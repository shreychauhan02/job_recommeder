"use client";

import { useCallback, useRef, useState } from "react";
import {
  MAX_PDF_BYTES,
  validateResumeFile,
  type ExplainResponse,
  type JobResult,
  type MatchMethod,
  type MatchResponse,
} from "@/lib/types";

const METHODS: { value: MatchMethod; label: string }[] = [
  { value: "tfidf", label: "TF-IDF" },
  { value: "embeddings", label: "Embeddings" },
  { value: "hybrid", label: "Hybrid" },
];

function ScoreBar({ score, label }: { score: number; label: string }) {
  const percent = Math.round(score * 100);
  return (
    <div>
      <div
        role="progressbar"
        aria-valuenow={percent}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`${label}: ${percent}%`}
        className="h-2.5 w-full overflow-hidden rounded-full bg-slate-200"
      >
        <div className="h-full rounded-full bg-emerald-500 transition-all" style={{ width: `${percent}%` }} />
      </div>
      <p className="mt-1 text-sm text-slate-600">
        {label}: <span className="font-semibold">{percent}%</span>
      </p>
    </div>
  );
}

function SkillChips({ skills, tone }: { skills: string[]; tone: "matched" | "missing" }) {
  if (skills.length === 0) return null;
  const classes =
    tone === "matched"
      ? "bg-emerald-100 text-emerald-800 border-emerald-300"
      : "bg-rose-100 text-rose-800 border-rose-300";
  return (
    <ul
      className="flex flex-wrap gap-1.5"
      aria-label={tone === "matched" ? "Matched skills" : "Missing skills"}
    >
      {skills.map((skill) => (
        <li key={skill} className={`rounded-full border px-2.5 py-0.5 text-xs font-medium ${classes}`}>
          {skill}
        </li>
      ))}
    </ul>
  );
}

function JobCard({ job, resumeSummary }: { job: JobResult; resumeSummary: string }) {
  const [explainState, setExplainState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);

  const onExplain = useCallback(async () => {
    setExplainState("loading");
    try {
      const response = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          resume_summary: resumeSummary,
          title: job.title,
          company: job.company,
          matched_skills: job.matched_skills,
          missing_skills: job.missing_skills,
        }),
      });
      const data = (await response.json()) as ExplainResponse & { error?: string };
      if (!response.ok) {
        setExplanation({ configured: false, message: data.error ?? "Explanation failed." });
        setExplainState("error");
        return;
      }
      setExplanation(data);
      setExplainState("done");
    } catch {
      setExplanation({ configured: false, message: "Could not reach the server." });
      setExplainState("error");
    }
  }, [job, resumeSummary]);

  return (
    <li className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="text-lg font-semibold">{job.title}</h3>
          <p className="text-sm text-slate-600">
            {job.company}
            {job.location ? ` · ${job.location}` : ""}
          </p>
        </div>
        {job.url ? (
          <a
            href={job.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-indigo-600 hover:underline"
          >
            View job →
          </a>
        ) : null}
      </div>

      <div className="mt-4">
        <ScoreBar score={job.score} label="Overall score" />
      </div>

      <div className="mt-4 space-y-3">
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Matched skills</p>
          <SkillChips skills={job.matched_skills} tone="matched" />
        </div>
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Missing skills</p>
          <SkillChips skills={job.missing_skills.slice(0, 10)} tone="missing" />
        </div>
      </div>

      <div className="mt-4">
        <button
          type="button"
          onClick={onExplain}
          disabled={explainState === "loading"}
          className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-60"
        >
          {explainState === "loading" ? "Thinking…" : "Why this match?"}
        </button>
        {(explainState === "done" || explainState === "error") && explanation ? (
          <p className="mt-3 rounded-lg bg-slate-100 p-3 text-sm text-slate-700" aria-live="polite">
            {explanation.configured
              ? explanation.message
              : "AI explanations are not configured yet (add LLM_API_KEY to the backend)."}
          </p>
        ) : null}
      </div>
    </li>
  );
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [resumeText, setResumeText] = useState<string>("");
  const [method, setMethod] = useState<MatchMethod>("hybrid");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<JobResult[] | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectFile = useCallback((selected: File | null) => {
    setError(null);
    setResults(null);
    if (!selected) {
      setFile(null);
      setResumeText("");
      return;
    }
    const problem = validateResumeFile(selected);
    if (problem) {
      setError(problem);
      setFile(null);
      return;
    }
    setFile(selected);
    selected.text().then((text) => setResumeText(text.slice(0, 400)));
  }, []);

  const onMatch = useCallback(async () => {
    if (!file) {
      setError("Please choose a resume PDF first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      form.append("method", method);
      const response = await fetch("/api/match", { method: "POST", body: form });
      const data = (await response.json()) as MatchResponse & { error?: string };
      if (!response.ok) {
        setError(data.error ?? `Matching failed (${response.status}).`);
        return;
      }
      setResults(data.results);
    } catch {
      setError("Could not reach the server. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, [file, method]);

  return (
    <main className="mx-auto w-full max-w-4xl flex-1 px-4 py-10 sm:px-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">Resume Job Matcher</h1>
        <p className="mt-2 text-slate-600">
          Upload your resume and rank 680 real Indeed job postings with TF-IDF, semantic embeddings, or hybrid scoring.
        </p>
      </header>

      <section aria-label="Resume upload" className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div
          onDragOver={(event) => {
            event.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragOver(false);
            selectFile(event.dataTransfer.files[0] ?? null);
          }}
          className={`flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
            dragOver ? "border-indigo-500 bg-indigo-50" : "border-slate-300"
          }`}
        >
          <p className="text-sm text-slate-600">
            Drag &amp; drop your resume PDF here (max {MAX_PDF_BYTES / 1024 / 1024} MB)
          </p>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
          >
            Browse files
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf,.pdf"
            className="sr-only"
            aria-label="Choose resume PDF"
            onChange={(event) => selectFile(event.target.files?.[0] ?? null)}
          />
          {file ? <p className="text-sm font-medium text-emerald-700">Selected: {file.name}</p> : null}
        </div>

        <fieldset className="mt-5">
          <legend className="text-sm font-semibold text-slate-700">Matching method</legend>
          <div className="mt-2 flex flex-wrap gap-2">
            {METHODS.map((option) => (
              <label
                key={option.value}
                className={`cursor-pointer rounded-lg border px-4 py-2 text-sm font-medium ${
                  method === option.value
                    ? "border-indigo-600 bg-indigo-600 text-white"
                    : "border-slate-300 bg-white text-slate-700 hover:border-indigo-400"
                }`}
              >
                <input
                  type="radio"
                  name="method"
                  value={option.value}
                  checked={method === option.value}
                  onChange={() => setMethod(option.value)}
                  className="sr-only"
                />
                {option.label}
              </label>
            ))}
          </div>
        </fieldset>

        <button
          type="button"
          onClick={onMatch}
          disabled={loading || !file}
          className="mt-5 w-full rounded-lg bg-indigo-600 px-4 py-2.5 font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Matching your resume…" : "Find matching jobs"}
        </button>

        {loading ? (
          <p role="status" aria-live="polite" className="mt-4 flex items-center gap-2 text-sm text-slate-600">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-indigo-600 border-t-transparent" />
            Ranking jobs, this can take a few seconds…
          </p>
        ) : null}
        {error ? (
          <p role="alert" className="mt-4 rounded-lg bg-rose-100 p-3 text-sm font-medium text-rose-800">
            {error}
          </p>
        ) : null}
      </section>

      {results ? (
        <section aria-label="Match results" className="mt-8">
          <h2 className="mb-4 text-xl font-semibold">Top matches ({results.length})</h2>
          {results.length === 0 ? (
            <p className="text-sm text-slate-600">No jobs matched. Try another method or check the backend index.</p>
          ) : (
            <ul className="space-y-4">
              {results.map((job) => (
                <JobCard
                  key={`${job.title}-${job.company}`}
                  job={job}
                  resumeSummary={resumeText || (file?.name ?? "")}
                />
              ))}
            </ul>
          )}
        </section>
      ) : null}
    </main>
  );
}
