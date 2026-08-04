import { create } from 'zustand';
import type { AgentMessage, PurchaseDraft } from '@/types';

function id() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

interface AppStore {
  sessionId: string;
  userId: string;
  draft?: PurchaseDraft;
  messages: AgentMessage[];
  setDraft: (draft: PurchaseDraft) => void;
  addMessage: (message: AgentMessage) => void;
  patchMessage: (messageId: string, patch: Partial<AgentMessage>) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  sessionId: `h5-${id()}`,
  userId: 'demo-user',
  messages: [
    {
      id: 'hello',
      role: 'assistant',
      content: '早上好，我是电影票智能体。你可以直接说想看的电影、时间、位置和人数。',
      cards: [],
    },
  ],
  setDraft: (draft) => set({ draft }),
  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  patchMessage: (messageId, patch) =>
    set((state) => ({
      messages: state.messages.map((message) =>
        message.id === messageId ? { ...message, ...patch } : message,
      ),
    })),
}));

export function newId() {
  return id();
}
