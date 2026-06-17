export interface ChatRequest {
  message: string;
  chat_history?: { role: string; content: string }[];
  model_choice?: string;
}

export interface ChatResponse {
  response: string;
  sources: { doc: string; page: number; chunk: string; score: number }[];
  confidence: number;
  query_type: string;
}

export interface DocumentInfo {
  id: number;
  name: string;
  company: string;
  year: number;
  type: string;
  pages: number;
  status: string;
  chunks: number;
  size: string;
}

export interface DashboardStats {
  documents_indexed: number;
  total_documents: number;
  total_chunks: number;
  vectorstore_ready: boolean;
  model_available: boolean;
}

export interface KnowledgeStats {
  total_vectors: number;
  collections: number;
  avg_similarity: number;
  index_size: string;
  embedding_model: string;
  dimensions: number;
  similarity_metric: string;
  top_k: number;
}

export interface SearchResult {
  id: number;
  source: string;
  page: number;
  score: number;
  preview: string;
  content: string;
}

export interface AgentStatus {
  name: string;
  status: string;
  model: string;
  latency: string;
  requests: number;
  accuracy: string;
  desc: string;
}

export interface RAGStatus {
  pipeline: string;
  agents: AgentStatus[];
  vectorstore_available: boolean;
  avg_response_time: string;
  daily_requests: number;
  token_usage: string;
}