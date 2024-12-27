package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
)

// program structs
type Embedding struct {
	Content   string
	Embedding []float32
}

// content hash -> Embedding struct
type EmbeddingMap map[string]Embedding

// globals
var (
	embeddings            EmbeddingMap
	embeddingsMutex       sync.RWMutex
	openAIAPIKey          string
	openAIEmbeddingsModel = "text-embedding-3-small"
	openAIChatModel       = "gpt-4o-mini"
	topK                  = 10
	dataDir               = "data"
	embeddingsFile        = dataDir + "/embeddings.json"
	jiraXMLFile           = dataDir + "/jira_tickets.xml"
)

// XML data
type Item struct {
	Title       string `xml:"title"`
	Description string `xml:"description"`
	Created     string `xml:"created"`
	Updated     string `xml:"updated"`
	Reporter    string `xml:"reporter"`
	Assignee    string `xml:"assignee"`
	Status      string `xml:"status"`
}

// Extract inner text from the provided HTML string
func htmlToText(html string) string {
	var result []rune
	inTag := false
	for _, r := range html {
		switch {
		case r == '<':
			inTag = true
		case r == '>':
			inTag = false
		default:
			if !inTag {
				result = append(result, r)
			}
		}
	}
	return string(result)
}

// Get string representation of XML Item
func (it Item) Content() string {
	description := strings.TrimSpace(it.Description)
	if description == "" {
		description = "<empty>"
	} else {
		description = htmlToText(description)
	}

	return fmt.Sprintf("Status: %s\nReporter: %s\nAssignee: %s\nCreated: %s\nUpdated: %s\nTitle: %s\nDescription:\n%s",
		it.Status, it.Reporter, it.Assignee, it.Created, it.Updated, it.Title, description)
}

// parseXML reads and parses the XML file into Items
func parseXML(filename string) ([]Item, error) {
	// internal structs for unmarshalling
	// (program only cares about []Item)
	type Channel struct {
		Items []Item `xml:"item"`
	}

	type JiraRSS struct {
		XMLName xml.Name `xml:"rss"`
		Channel Channel  `xml:"channel"`
	}

	xmlFile, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error opening file %v: %w", filename, err)
	}
	defer xmlFile.Close()

	byteValue, _ := io.ReadAll(xmlFile)

	var rss JiraRSS
	err = xml.Unmarshal(byteValue, &rss)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling XML: %w", err)
	}

	return rss.Channel.Items, nil
}

// region OpenAI API
// Call OpenAI embedding API endpoint to compute embedding of the input `text` string
func callOpenAIEmbeddingEndpoint(text string) ([]float32, error) {
	url := "https://api.openai.com/v1/embeddings"
	payload := map[string]interface{}{
		"input":           text,
		"model":           openAIEmbeddingsModel,
		"encoding_format": "float",
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("error marshalling embedding request payload: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("error creating embedding request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+openAIAPIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error sending embedding request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedding API error: %s", string(bodyBytes))
	}

	// JSON deserialisation
	var embeddingResponse struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}

	err = json.NewDecoder(resp.Body).Decode(&embeddingResponse)
	if err != nil {
		return nil, fmt.Errorf("error deserialising embedding response: %w", err)
	}

	if len(embeddingResponse.Data) == 0 {
		return nil, fmt.Errorf("no embedding data returned")
	}

	return embeddingResponse.Data[0].Embedding, nil
}

// Call OpenAI completions API endpoint
func callOpenAICompletionsEndpoint(systemPrompt, prompt string) (string, error) {
	url := "https://api.openai.com/v1/chat/completions"
	payload := map[string]interface{}{
		"model": openAIChatModel,
		"messages": []map[string]string{
			{
				"role":    "system",
				"content": systemPrompt,
			},
			{
				"role":    "user",
				"content": prompt,
			},
		},
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("error marshalling chat completion request payload: %w", err)
	}

	req, err := http.NewRequest("POST", url, bytes.NewBuffer(payloadBytes))
	if err != nil {
		return "", fmt.Errorf("error creating chat completion request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+openAIAPIKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("error sending chat completion request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("chat completion API error: %s", string(bodyBytes))
	}

	var completionResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
				Role    string `json:"role"`
			} `json:"message"`
			Index        int         `json:"index"`
			FinishReason string      `json:"finish_reason"`
			Logprobs     interface{} `json:"logprobs"`
		} `json:"choices"`
	}

	err = json.NewDecoder(resp.Body).Decode(&completionResponse)
	if err != nil {
		return "", fmt.Errorf("error decoding chat completion response: %w", err)
	}

	if len(completionResponse.Choices) == 0 {
		return "", fmt.Errorf("no completion choices returned")
	}

	return completionResponse.Choices[0].Message.Content, nil
}

// endregion

// region embedding
// Compute a sha256 hash of the input text string
func hashContent(text string) string {
	h := sha256.New()
	h.Write([]byte(text))
	return fmt.Sprintf("%x", h.Sum(nil))
}

// Compute the embedding for the input `text` using OpenAI, cached with the filesystem
func getEmbedding(text string) ([]float32, error) {
	hash := hashContent(text)

	// Check cache
	embeddingsMutex.RLock()
	if emb, exists := embeddings[hash]; exists {
		embeddingsMutex.RUnlock()
		return emb.Embedding, nil
	}
	embeddingsMutex.RUnlock()

	// Cache miss, call OpenAI
	// NOTE: calling individually for now (rather than batching API calls)
	embedding, err := callOpenAIEmbeddingEndpoint(text)
	if err != nil {
		return nil, err
	}

	// Store new embedding
	embeddingsMutex.Lock()
	embeddings[hash] = Embedding{
		Content:   text,
		Embedding: embedding,
	}
	embeddingsMutex.Unlock()

	return embedding, nil
}

// Compute the `embeddings` embedding map using `getEmbedding` from the provided XML `items`
func computeEmbeddings(items []Item) error {
	var wg sync.WaitGroup
	sem := make(chan struct{}, 5) // Limit concurrent API calls

	for _, item := range items {
		wg.Add(1)
		sem <- struct{}{}
		go func(it Item) {
			defer wg.Done()
			defer func() { <-sem }()
			content := it.Content()
			hash := hashContent(content)

			embeddingsMutex.RLock()
			_, exists := embeddings[hash]
			embeddingsMutex.RUnlock()

			if exists {
				// Embedding already exists, skip computation
				return
			}

			embed, err := getEmbedding(content)
			if err != nil {
				log.Printf("Error getting embedding for item '%s': %v", it.Title, err)
				return
			}

			embeddingsMutex.Lock()
			embeddings[hash] = Embedding{
				Content:   content,
				Embedding: embed,
			}
			embeddingsMutex.Unlock()

			// Save embeddings to file after each new embedding is added
			err = saveEmbeddings()
			if err != nil {
				log.Printf("Error saving embeddings: %v", err)
			}
		}(item)
	}

	wg.Wait()
	return nil
}

// region loading and saving embeddings to/from disk
// Load embeddings from disk file into memory
func loadEmbeddings() error {
	embeddingsMutex.Lock()
	defer embeddingsMutex.Unlock()

	file, err := os.Open(embeddingsFile)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	err = decoder.Decode(&embeddings)
	if err != nil {
		return fmt.Errorf("error decoding embeddings file: %w", err)
	}

	return nil
}

// Save embeddings from memory to disk
func saveEmbeddings() error {
	embeddingsMutex.Lock()
	defer embeddingsMutex.Unlock()

	tempFile := embeddingsFile + ".tmp"
	file, err := os.Create(tempFile)
	if err != nil {
		return fmt.Errorf("error creating temporary embeddings file: %w", err)
	}

	encoder := json.NewEncoder(file)
	err = encoder.Encode(embeddings)
	if err != nil {
		file.Close()
		return fmt.Errorf("error encoding embeddings to file: %w", err)
	}

	file.Close()

	// Replace the old embeddings file with the new one atomically
	err = os.Rename(tempFile, embeddingsFile)
	if err != nil {
		return fmt.Errorf("error renaming temporary embeddings file: %w", err)
	}

	return nil
}

// endregion

// endregion

// region vector search

// TODO: change to content string
type SearchResult struct {
	Content string
	Score   float32
}

type ByScore []SearchResult

func (a ByScore) Len() int           { return len(a) }
func (a ByScore) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByScore) Less(i, j int) bool { return a[i].Score > a[j].Score }

// cosine similarity between two equal-length vectors a and b
func cosineSimilarity(a, b []float32) float32 {
	var sumProduct, sumASq, sumBSq float32
	for i := range a {
		sumProduct += a[i] * b[i]
		sumASq += a[i] * a[i]
		sumBSq += b[i] * b[i]
	}
	if sumASq == 0 || sumBSq == 0 {
		return 0
	}
	return sumProduct / (float32(math.Sqrt(float64(sumASq))) * float32(math.Sqrt(float64(sumBSq))))
}

// findTopKSimilar finds the top K items most similar to the query embedding
func findTopKSimilar(queryEmbedding []float32, k int) ([]SearchResult, error) {
	embeddingsMutex.RLock()
	defer embeddingsMutex.RUnlock()

	var results []SearchResult
	for _, emb := range embeddings {
		score := cosineSimilarity(queryEmbedding, emb.Embedding)
		results = append(results, SearchResult{
			Content: emb.Content,
			Score:   score,
		})
	}

	// Sort results by score descending
	sort.Sort(ByScore(results))

	// Return top K results
	if len(results) < k {
		k = len(results)
	}
	return results[:k], nil
}

// endregion

// buildContext builds the context text from the top items
func buildContext(topItems []SearchResult) string {
	var sb strings.Builder
	for _, item := range topItems {
		sb.WriteString(fmt.Sprintf("-- Jira Ticket --\n%s\n\n", item.Content))
	}
	return sb.String()
}

// Gin handler for the /query endpoint
func handleQuery(c *gin.Context) {
	var request struct {
		Query string `json:"query"`
	}
	if err := c.BindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}
	if request.Query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Query cannot be empty"})
		return
	}

	// Compute embedding
	queryEmbedding, err := callOpenAIEmbeddingEndpoint(request.Query)
	if err != nil {
		log.Printf("Error getting embedding for query: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error processing query"})
		return
	}

	// Find top k similar items
	topItems, err := findTopKSimilar(queryEmbedding, topK)
	if err != nil {
		log.Printf("Error finding top K similar items: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error processing query"})
		return
	}

	// LLM input construction and evaluation
	// Construct the prompt
	systemPrompt := "You are a helpful conversational assistant who answers questions based on provided Jira tickets."

	contextText := buildContext(topItems)
	prompt := fmt.Sprintf(systemPrompt+" "+
		"Using the following retrieved Jira tickets as context:\n\n%s\n\nAnswer the following question:\n%s\n\n"+
		"Only reference relevant Jira tickets in your answer. "+
		"When referencing specific Jira tickets in your answer, reference its full ticket name and title in the format: [KEY] <TITLE>.\n"+
		"When it makes sense to do so, provide a bullet point list of tickets with an explanation for each point, with the following "+
		"format for each bullet point: - [KEY] <TITLE>  --  <EXPLANATION>\n"+
		// -- avoid saying 'from the provided Jira tickets': --
		"Present your answer naturally and conversationally as a human would, without implying any automated process behind the formulation of your answer. "+
		"If applicable, present yourself as having processed the information entirely yourself instead of implying that Jira tickets were provided to you as context.",
		contextText, request.Query)

	log.Printf("Constructed prompt:\n%s\n", prompt)

	// Call ChatGPT API
	answer, err := callOpenAICompletionsEndpoint(systemPrompt, prompt)
	if err != nil {
		log.Printf("Error getting completion: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error processing query"})
		return
	}

	// Return the answer to the user
	c.JSON(http.StatusOK, gin.H{"answer": answer})
}

func main() {
	// load dotenv
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	// OpenAI API key
	openAIAPIKey = os.Getenv("OPENAI_API_KEY")
	if openAIAPIKey == "" {
		log.Fatal("Please set the OPENAI_API_KEY environment variable")
	}

	// Init embeddings map
	embeddings = make(EmbeddingMap)

	// Init data directory
	if _, err := os.Stat(dataDir); os.IsNotExist(err) {
		err := os.Mkdir(dataDir, 0755)
		if err != nil {
			log.Fatalf("Failed to create data directory: %v", err)
		}
	}

	// Load existing embeddings from disk
	err = loadEmbeddings()
	if err != nil {
		log.Printf("No existing embeddings found or error loading embeddings: %v", err)
	}

	// Parse XML file from disk
	items, err := parseXML(jiraXMLFile)
	if err != nil {
		log.Fatalf("Error parsing XML: %v", err)
	}

	// Compute embeddings for all items
	err = computeEmbeddings(items)
	if err != nil {
		log.Fatalf("Error computing embeddings: %v", err)
	}

	// Save embeddings to disk
	err = saveEmbeddings()
	if err != nil {
		log.Fatalf("Error saving embeddings: %v", err)
	}

	// Gin router
	router := gin.Default()
	router.POST("/query", handleQuery)
	router.Run(":8080")
}
