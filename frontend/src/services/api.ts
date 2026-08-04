import type { AgentCard, ApiResponse, Movie, Order, PurchaseDraft, Seat, Showtime } from '@/types';

export const API_BASE = (process.env.UMI_APP_API_BASE || 'http://127.0.0.1:8001').replace(/\/$/, '');

async function get<T>(path: string, params?: Record<string, unknown>) {
  const url = new URL(`${API_BASE}${path}`);
  Object.entries(params || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      url.searchParams.set(key, String(value));
    }
  });
  const response = await fetch(url);
  const payload = (await response.json()) as ApiResponse<T>;
  if (payload.code !== 0) throw new Error(payload.message || '请求失败');
  return payload.data;
}

async function post<T>(path: string, body?: Record<string, unknown>) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const payload = (await response.json()) as ApiResponse<T>;
  if (payload.code !== 0) throw new Error(payload.message || '请求失败');
  return payload.data;
}

export function getActiveDraft() {
  return get<PurchaseDraft>('/api/v1/drafts/active');
}

export function updateDraft(body: Partial<PurchaseDraft>) {
  return post<PurchaseDraft>('/api/v1/drafts', body);
}

export function listMovies() {
  return get<{ movies: Movie[] }>('/api/v1/movies');
}

export function listShowtimes(params?: Record<string, unknown>) {
  return get<{ showtimes: Showtime[] }>('/api/v1/showtimes', params);
}

export function getSeats(showtimeId: string) {
  return get<{ showtimeId: string; seats: Seat[] }>(`/api/v1/showtimes/${showtimeId}/seats`);
}

export function createOrder(body: {
  draftId?: number;
  showtimeId: string;
  seatIds: string[];
  ticketCount?: number;
  userId?: string;
}) {
  return post<Order>('/api/v1/orders', body);
}

export function payOrder(orderId: string, idempotencyKey: string) {
  return post<Order>(`/api/v1/orders/${orderId}/pay`, { idempotencyKey });
}

interface StreamCallbacks {
  onThinking: (text: string) => void;
  onMessage: (text: string) => void;
  onCard: (card: AgentCard) => void;
  onError: (text: string) => void;
  onDone: () => void;
}

function parseBlocks(buffer: string) {
  const blocks = buffer.split('\n\n');
  const rest = blocks.pop() || '';
  return {
    rest,
    events: blocks.map((block) => {
      const event = block.split('\n').find((line) => line.startsWith('event:'))?.replace('event:', '').trim();
      const data = block
        .split('\n')
        .filter((line) => line.startsWith('data:'))
        .map((line) => line.replace('data:', '').trim())
        .join('\n');
      return { event, data };
    }).filter((item) => item.event && item.data),
  };
}

export async function streamAgent(
  body: {
    sessionId: string;
    draftId?: number;
    message: string;
    event?: string;
    userId?: string;
    payload?: Record<string, unknown>;
  },
  callbacks: StreamCallbacks,
) {
  const response = await fetch(`${API_BASE}/api/agent/chat/stream`, {
    method: 'POST',
    headers: { Accept: 'text/event-stream', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok || !response.body) {
    callbacks.onError(`Agent 请求失败：${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseBlocks(buffer);
    buffer = parsed.rest;
    parsed.events.forEach(({ event, data }) => {
      const json = JSON.parse(data);
      if (event === 'thinking') callbacks.onThinking(json.message || '');
      if (event === 'message') callbacks.onMessage(json.content || '');
      if (event === 'card') callbacks.onCard(json.data);
      if (event === 'error') callbacks.onError(json.message || 'Agent 处理失败');
      if (event === 'done') callbacks.onDone();
    });
  }
}
