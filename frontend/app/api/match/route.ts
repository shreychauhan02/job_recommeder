import { NextRequest, NextResponse } from "next/server";
import { MAX_PDF_BYTES, type ApiError, type MatchMethod } from "@/lib/types";

const ALLOWED_METHODS: MatchMethod[] = ["tfidf", "embeddings", "hybrid"];

export async function POST(request: NextRequest): Promise<NextResponse<ApiError | Response>> {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return NextResponse.json({ error: "Backend is not configured (BACKEND_URL missing)." }, { status: 500 });
  }

  let formData: FormData;
  try {
    formData = await request.formData();
  } catch {
    return NextResponse.json({ error: "Malformed request; expected multipart form data." }, { status: 400 });
  }

  const file = formData.get("file");
  const method = (formData.get("method") ?? "hybrid") as MatchMethod;

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "No resume file provided." }, { status: 400 });
  }
  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    return NextResponse.json({ error: "Only PDF files are accepted." }, { status: 400 });
  }
  if (file.size > MAX_PDF_BYTES) {
    return NextResponse.json({ error: "File is larger than 5 MB." }, { status: 413 });
  }
  if (!ALLOWED_METHODS.includes(method)) {
    return NextResponse.json({ error: `Unknown method '${method}'.` }, { status: 400 });
  }

  const proxyForm = new FormData();
  proxyForm.append("file", file, file.name);
  proxyForm.append("method", method);
  proxyForm.append("top_n", "20");

  try {
    const response = await fetch(`${backendUrl}/match`, { method: "POST", body: proxyForm, cache: "no-store" });
    const json = await response.json();
    if (!response.ok) {
      return NextResponse.json({ error: String(json.detail ?? `Backend error ${response.status}`) }, { status: response.status });
    }
    return NextResponse.json(json);
  } catch {
    return NextResponse.json({ error: "Could not reach the matching backend. Is FastAPI running?" }, { status: 502 });
  }
}
