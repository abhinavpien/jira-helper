package main

import (
	"os"
	"reflect"
	"testing"
)

func TestHtmlToText(t *testing.T) {
	tests := []struct {
		html     string
		expected string
	}{
		{"<p>This is <b>bold</b> text.</p>", "This is bold text."},
		{"<div><span>Nested <i>tags</i></span></div>", "Nested tags"},
		{"No tags here.", "No tags here."},
		{"<a href='url'>Link</a>", "Link"},
		{"<p>Special characters &amp; entities</p>", "Special characters &amp; entities"},
	}

	for _, test := range tests {
		result := htmlToText(test.html)
		if result != test.expected {
			t.Errorf("htmlToText(%q) = %q; want %q", test.html, result, test.expected)
		}
	}
}

func TestHashContent(t *testing.T) {
	s1 := "Abhinav"
	s2 := "Bhandari"

	h1 := hashContent(s1)
	h2 := hashContent(s2)
	h3 := hashContent(s1)

	if h1 == h2 {
		t.Errorf("hashContent(%q) and hashContent(%q) should not be equal", s1, s2)
	}

	if h1 != h3 {
		t.Errorf("hashContent(%q) and hashContent(%q) should be equal", s1, s1)
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		a, b     []float32
		expected float32
	}{
		{[]float32{1, 0}, []float32{1, 0}, 1},
		{[]float32{1, 0}, []float32{0, 1}, 0},
		{[]float32{1, 2}, []float32{2, 4}, 1},
		{[]float32{1, 0}, []float32{0, 2}, 0},
	}

	for _, test := range tests {
		result := cosineSimilarity(test.a, test.b)
		if !almostEqual(result, test.expected) {
			t.Errorf("cosineSimilarity(%v, %v) = %v; want %v", test.a, test.b, result, test.expected)
		}
	}
}

func almostEqual(a, b float32) bool {
	const epsilon = 1e-5
	return (a-b) < epsilon && (b-a) < epsilon
}

func TestFindTopKSimilar(t *testing.T) {
	// Prepare mock embeddings
	embeddings = EmbeddingMap{
		"hash1": Embedding{
			Content:   "Content 1",
			Embedding: []float32{1, 0, 0},
		},
		"hash2": Embedding{
			Content:   "Content 2",
			Embedding: []float32{0, 1, 0},
		},
		"hash3": Embedding{
			Content:   "Content 3",
			Embedding: []float32{0, 0, 1},
		},
	}

	// Query embedding
	queryEmbedding := []float32{1, 0, 0}

	results, err := findTopKSimilar(queryEmbedding, 2)
	if err != nil {
		t.Fatalf("findTopKSimilar failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}

	expectedContents := []string{"Content 1", "Content 2"}
	for i, result := range results {
		if result.Content != expectedContents[i] {
			t.Errorf("Result %d content = %q; want %q", i, result.Content, expectedContents[i])
		}
	}
}

func TestParseXML(t *testing.T) {
	xmlContent := `
	<rss>
		<channel>
			<item>
				<title>Issue 1</title>
				<description>Description of issue 1</description>
				<created>2023-01-01</created>
				<updated>2023-01-02</updated>
				<reporter>Alice</reporter>
				<assignee>Bob</assignee>
				<status>Open</status>
			</item>
			<item>
				<title>Issue 2</title>
				<description>Description of issue 2</description>
				<created>2023-02-01</created>
				<updated>2023-02-02</updated>
				<reporter>Charlie</reporter>
				<assignee>Dave</assignee>
				<status>Closed</status>
			</item>
		</channel>
	</rss>
	`
	filename := "test_jira.xml"
	err := os.WriteFile(filename, []byte(xmlContent), 0644)
	if err != nil {
		t.Fatalf("Failed to write temporary XML file: %v", err)
	}
	defer os.Remove(filename)

	items, err := parseXML(filename)
	if err != nil {
		t.Fatalf("parseXML failed: %v", err)
	}

	if len(items) != 2 {
		t.Errorf("Expected 2 items, got %d", len(items))
	}

	expectedItem := Item{
		Title:       "Issue 1",
		Description: "Description of issue 1",
		Created:     "2023-01-01",
		Updated:     "2023-01-02",
		Reporter:    "Alice",
		Assignee:    "Bob",
		Status:      "Open",
	}

	if !reflect.DeepEqual(items[0], expectedItem) {
		t.Errorf("First item mismatch.\nGot: %+v\nWant: %+v", items[0], expectedItem)
	}
}
