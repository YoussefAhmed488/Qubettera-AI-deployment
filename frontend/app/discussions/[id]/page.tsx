"use client";

import { use } from "react";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useDiscussion } from "@/contexts/DiscussionContext";
import { ChatView } from "@/components/discussion/ChatView";
import { ParticipantList } from "@/components/discussion/ParticipantList";
import { EvidencePanel } from "@/components/discussion/EvidencePanel";
import { BriefPanel } from "@/components/discussion/BriefPanel";
import { StatusPill } from "@/components/layout/StatusPill";
import { Button } from "@/components/ui/Button";
import { InlineError } from "@/components/ui/ErrorState";
import { toast } from "@/components/ui/Toast";
import { BarChart2, ArrowLeft } from "lucide-react";
import type { DiscussionDetail, EvidenceItem } from "@/types";
import { truncate } from "@/lib/format";

interface PageProps {
  params: Promise<{ id: string }>;
}

export default function DiscussionPage({ params }: PageProps) {
  const { id } = use(params);
  const { state, startDiscussion } = useDiscussion();
  const [replay, setReplay] = useState<{
    id: string;
    data?: DiscussionDetail;
    error?: string;
  } | null>(null);
  const replayData = replay?.id === id ? replay.data : undefined;
  const replayError = replay?.id === id ? replay.error : undefined;
  const replayLoading = replay?.id !== id;

  // A discussion is "live" when the React context has an active SSE stream:
  //   a) Context ID matches URL and stream isn't idle, OR
  //   b) Stream is active but discussionId not yet confirmed (race window
  //      between router.push() and the first stream_started event).
  const isLive =
    (state.discussionId === id && state.status !== "idle") ||
    (state.status === "streaming" && state.discussionId === null);

  // The replay fetch short-circuits while `isLive` is true, so `replay` stays
  // null and `replayLoading` would remain true forever. Scope the replay
  // loading/error gates to non-live mode, otherwise a live stream renders as
  // "Loading transcript…" for the whole discussion.
  const showReplayLoading = !isLive && replayLoading;
  const showReplayError = !isLive && !replayLoading && replayError !== undefined;

  // ── Replay (no live SSE context) ─────────────────────────────────────────
  // Poll active transcripts until the backend reports a terminal state.
  // Slow model turns can take minutes without producing a new message.
  useEffect(() => {
    if (isLive) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function loadReplay() {
      try {
        const response = await fetch(`/api/discussions/${encodeURIComponent(id)}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data: DiscussionDetail = await response.json();
        if (controller.signal.aborted) return;
        setReplay({ id, data });
        if (data.status === "running") timer = setTimeout(loadReplay, 3_000);
      } catch (error) {
        if (controller.signal.aborted) return;
        setReplay((previous) => ({
          id,
          data: previous?.id === id ? previous.data : undefined,
          error: error instanceof Error ? error.message : "Failed to load discussion.",
        }));
        timer = setTimeout(loadReplay, 3_000);
      }
    }
    void loadReplay();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [id, isLive]);

  // Toast on discussion completion.
  useEffect(() => {
    if (state.status === "done") {
      toast("success", "Discussion complete. Preparing analytics…");
      // Prefetch analytics in the background so it is cached when the user navigates.
      void fetch(`/api/week4/analytics/${state.discussionId}`, { keepalive: true });
    }
  }, [state.status, state.discussionId]);

  // Determine what data to display.
  const messages = isLive ? state.messages : (replayData?.messages ?? []);
  const participants = isLive
    ? state.participants
    : Object.fromEntries(
        (replayData?.config.participant_ids ?? []).map((pid) => [
          pid,
          { id: pid, status: "spoke" as const },
        ]),
      );

  // Build agent name map from messages (since we may not have /api/personas loaded here).
  const agentNames: Record<string, string> = {};
  for (const msg of messages) {
    if (!agentNames[msg.sender_id]) {
      // Derive a human-readable name from the ID: dr_aris -> Dr Aris.
      agentNames[msg.sender_id] = msg.sender_id
        .split("_")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
    }
  }

  // Latest turn evidence for sidebar.
  const lastMessage = messages[messages.length - 1];
  const latestEvidence: EvidenceItem[] = lastMessage?.evidence ?? [];

  // Status derived from live or replay context.
  // replayRunning = discussion is ongoing but this tab has no SSE connection
  // (page-refresh / History-tab link); we are polling REST for updates.
  const replayRunning = !isLive && replayData?.status === "running";
  const streamStatus = isLive ? state.status
    : replayData?.status === "completed" ? "done"
    : replayData?.status === "failed" ? "error" : "idle";
  const topicLabel = isLive
    ? (state.config?.brief.objective ?? "Discussion")
    : (replayData?.topic ?? id);

  return (
    <div className="flex h-[calc(100dvh-4rem)] flex-col overflow-hidden">
      {/* Page header */}
      <div className="flex shrink-0 items-center justify-between border-b border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-4 py-3">
        <div className="flex items-center gap-3 min-w-0">
          <Link
            href="/discussions"
            className="shrink-0 rounded-md p-1 text-slate-400 hover:text-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            aria-label="Back to discussions"
          >
            <ArrowLeft className="h-5 w-5" aria-hidden="true" />
          </Link>
          <div className="min-w-0">
            <h1 className="text-sm font-semibold text-slate-900 dark:text-slate-100 truncate">
              {truncate(topicLabel, 80)}
            </h1>
            <p className="text-xs font-mono text-slate-400 mt-0.5">{id.slice(0, 8)}…</p>
          </div>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <StatusPill
            status={
              streamStatus === "streaming" || replayRunning
                ? "streaming"
                : streamStatus === "done" && !replayRunning
                  ? "complete"
                  : streamStatus === "error"
                    ? "error"
                    : "idle"
            }
            label={
              streamStatus === "streaming"
                ? `Streaming · R${Math.max(...messages.map((m) => m.round_number), 0)}/${state.config?.num_rounds ?? "?"}`
                : replayRunning
                  ? `Updating… · ${messages.length} turns`
                  : streamStatus === "done"
                    ? "Complete"
                    : streamStatus === "error"
                      ? "Failed"
                      : "Replay"
            }
          />

          {(streamStatus === "done" || replayData) && (
            <Link href={`/analytics/${id}`}>
              <Button size="sm" variant="primary" id={`view-analytics-${id}`}>
                <BarChart2 className="h-4 w-4" aria-hidden="true" />
                View Analytics
              </Button>
            </Link>
          )}
        </div>
      </div>

      {/* Brief panel */}
      {(state.brief || replayData?.config.brief) && (
        <div className="shrink-0 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700">
          <BriefPanel
            brief={(state.brief ?? replayData!.config.brief)!}
            participantIds={state.config?.participant_ids ?? replayData?.config.participant_ids ?? []}
            agentNames={agentNames}
            numRounds={state.config?.num_rounds ?? replayData?.config.num_rounds ?? 0}
            discussionId={id}
            createdAt={replayData?.created_at ?? new Date().toISOString()}
          />
        </div>
      )}

      {/* Error banner */}
      {state.status === "error" && (
        <div className="shrink-0 px-4 pt-3">
          <InlineError
            message={state.error ?? "Stream failed."}
            onRetry={() => {
              if (state.config) {
                startDiscussion({
                  topic: state.config.brief.objective,
                  participant_ids: state.config.participant_ids,
                  num_rounds: state.config.num_rounds,
                  mode: "live",
                });
              }
            }}
          />
        </div>
      )}

      {/* Main layout — chat + sidebar */}
      <div className="flex flex-1 overflow-hidden">
        {/* Chat column */}
        <div className="flex flex-1 flex-col overflow-hidden">
          {showReplayLoading ? (
            <div className="flex flex-1 items-center justify-center text-sm text-slate-400">
              Loading transcript…
            </div>
          ) : showReplayError ? (
            <div className="p-6">
              <InlineError message={replayError!} />
            </div>
          ) : (
            <ChatView
              messages={messages}
              agentNames={agentNames}
              streaming={streamStatus === "streaming"}
              streamingAgentId={isLive ? state.streamingAgentId : null}
              streamingText={isLive ? state.streamingText : ""}
            />
          )}
        </div>

        {/* Sidebar */}
        <aside
          className="hidden md:flex w-64 shrink-0 flex-col border-l border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 overflow-y-auto"
          aria-label="Discussion sidebar"
        >
          <div className="p-4 space-y-6">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500 mb-3">
                Participants
              </h2>
              <ParticipantList
                participants={participants}
                agentNames={agentNames}
              />
            </div>

            <hr className="border-slate-200 dark:border-slate-700" />

            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500 mb-3">
                Evidence — Last Turn
              </h2>
              <EvidencePanel evidence={latestEvidence} />
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
