export type MatchMethod = "tfidf" | "embeddings" | "hybrid";

export interface JobResult {
  title: string;
  company: string;
  location: string;
  url: string;
  score: number;
  semantic_score: number | null;
  skill_score: number | null;
  matched_skills: string[];
  missing_skills: string[];
}

export interface MatchResponse {
  results: JobResult[];
  method: MatchMethod;
  total_jobs: number;
}

export interface ExplainRequest {
  resume_summary: string;
  title: string;
  company: string;
  matched_skills: string[];
  missing_skills: string[];
}

export interface ExplainResponse {
  configured: boolean;
  message: string;
}

export interface ApiError {
  error: string;
}

export const MAX_PDF_BYTES = 5 * 1024 * 1024;

export function validateResumeFile(file: File): string | null {
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    return "Only PDF files are accepted.";
  }
  if (file.size > MAX_PDF_BYTES) {
    return "File is larger than 5 MB.";
  }
  if (file.size === 0) {
    return "File is empty.";
  }
  return null;
}
