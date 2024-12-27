package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

var endpoint = "http://localhost:8080/query"

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run . \"<Your question>\"")
		return
	}

	query := strings.Join(os.Args[1:], " ")

	// Request
	payload := map[string]string{"query": query}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		fmt.Println("Error marshaling payload:", err)
		return
	}

	resp, err := http.Post(endpoint, "application/json", bytes.NewBuffer(payloadBytes))
	if err != nil {
		fmt.Println("Error making POST request:", err)
		return
	}
	defer resp.Body.Close()

	// Read response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Println("Error reading response body:", err)
		return
	}

	var response map[string]string
	if err := json.Unmarshal(body, &response); err != nil {
		fmt.Println("Error unmarshaling response:", err)
		return
	}

	// Display
	if answer, ok := response["answer"]; ok {
		fmt.Println(answer)
	} else if errorMsg, ok := response["error"]; ok {
		fmt.Println("Error:", errorMsg)
	} else {
		fmt.Println("Unknown response")
	}
}
