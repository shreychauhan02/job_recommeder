import { NextRequest, NextResponse } from "next/server";
import type { ApiError, ExplainRequest } from "@/lib/types";

export async function POST(request: NextRequest): Promise<NextResponse<ApiError | object>> {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    return NextResponse.json({ error: "Backend is not configured (BACKEND_URL missing)." }, { status: 500 });
  }

  let payload: ExplainRequest;
  try {
    payload = (await request.json()) as ExplainRequest;
  } catch {
    return NextResponse.json({ error: "Malformed JSON body." }, { status: 400 });
  }

  if (typeof payload.resume_summary !== "string" || payload.resume_summary.trim().length === 0 || typeof payload.title !== "string") {
    return NextResponse.json({ error: "resume_summary and title are required." }, { status: 400 });
  }

  const body: ExplainRequest = {
    resume_summary: payload.resume_summary.slice(0, 2000),
    title: payload.title,
    company: payload.company ?? "",
    matched_skills: Array.isArray(payload.matched_skills) ? payload.matched_skills.slice(0, 50) : [],
    missing_skills: Array.isArray(payload.missing_skills) ? payload.missing_skills.slice(0, 50) : [],
  };

  try {
    const response = await fetch(`${backendUrl}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    const json = (await response.json()) as object;
    if (!response.ok) {
      return NextResponse.json({ error: `Backend error ${response.status}` }, { status: response.status });
    }
    return NextResponse.json(json);
  } catch {
    return NextResponse.json({ error: "Could not reach the matching backend." }, { status: 502 });
  }
}
