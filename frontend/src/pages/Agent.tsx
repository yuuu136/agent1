import { useMemo, useState } from 'react';
import PrototypeShell from '@/components/PrototypeShell';
import SeatGrid from '@/components/SeatGrid';
import StepRail from '@/components/StepRail';
import { streamAgent } from '@/services/api';
import { newId, useAppStore } from '@/store/appStore';
import type { AgentCard } from '@/types';
import styles from './Agent.module.css';

function typeLabel(type: string) {
  return type.replace('_LIST', '').replace('_', ' ');
}

function metaValue(value: unknown) {
  if (value === null || value === undefined || typeof value === 'object') return '';
  return String(value);
}

function AgentCards({
  cards,
  disabled,
  onAction,
}: {
  cards?: AgentCard[];
  disabled?: boolean;
  onAction: (event: string, message: string, payload?: Record<string, unknown>) => void;
}) {
  if (!cards?.length) return null;

  return (
    <div className={styles.cards}>
      {cards.map((card, index) => {
        if (card.type === 'SEAT_MAP') {
          return (
            <SeatGrid
              key={`${card.type}-${card.id}-${index}`}
              seats={card.seats || []}
              limit={2}
              onConfirm={(seatIds) => onAction('select_seats', `选择座位 ${seatIds.join('、')}`, { showtimeId: card.id, seatIds })}
            />
          );
        }

        const meta = Object.entries(card.meta || {})
          .map(([key, value]) => [key, metaValue(value)] as const)
          .filter(([, value]) => value);
        return (
          <article className={styles.agentCard} key={`${card.type}-${card.id}-${index}`}>
            <div className={styles.cardTop}>
              <strong>{card.title || card.id || '候选项'}</strong>
              <span>{typeLabel(card.type)}</span>
            </div>
            {card.subtitle ? <p>{card.subtitle}</p> : null}
            {meta.length ? (
              <div className={styles.meta}>
                {meta.map(([key, value]) => (
                  <em key={key}>{key}: {value}</em>
                ))}
              </div>
            ) : null}
            {card.actions?.length ? (
              <div className={styles.actions}>
                {card.actions.map((action) => (
                  <button
                    key={action.event}
                    type="button"
                    disabled={disabled}
                    onClick={() => onAction(action.event, action.label, action.payload || card.payload || { id: card.id })}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}

export default function AgentPage() {
  const { sessionId, userId, draft, messages, addMessage, patchMessage } = useAppStore();
  const [input, setInput] = useState('');
  const [running, setRunning] = useState(false);
  const draftId = useMemo(() => {
    const query = new URLSearchParams(window.location.hash.split('?')[1] || '');
    const value = Number(query.get('draftId'));
    return Number.isFinite(value) && value > 0 ? value : draft?.draftId;
  }, [draft?.draftId]);

  async function send(message: string, event?: string, payload?: Record<string, unknown>) {
    const text = message.trim();
    if (!text || running) return;

    const assistantId = newId();
    const cards: AgentCard[] = [];
    addMessage({ id: newId(), role: 'user', content: text });
    addMessage({ id: assistantId, role: 'assistant', content: '', status: '正在连接 Agent', cards: [] });
    setInput('');
    setRunning(true);

    try {
      await streamAgent(
        { sessionId, draftId, message: text, event, userId, payload },
        {
          onThinking: (status) => patchMessage(assistantId, { status }),
          onMessage: (content) => patchMessage(assistantId, { content, status: undefined }),
          onCard: (card) => {
            cards.push(card);
            patchMessage(assistantId, { cards: [...cards] });
          },
          onError: (content) => patchMessage(assistantId, { content, status: undefined }),
          onDone: () => setRunning(false),
        },
      );
    } catch {
      patchMessage(assistantId, { content: '我换一种方式继续帮你处理。', status: undefined });
    } finally {
      setRunning(false);
    }
  }

  return (
    <PrototypeShell mode="agent">
      <StepRail active={draft?.showtimeId ? 2 : 0} />
      <section className={styles.welcome}>
        <span>AI 购票首页</span>
        <h1>一句话买电影票</h1>
        <p>支持自然语言、卡片选择、自动跳步和异常替代方案。</p>
        <div className={styles.suggestions}>
          {['今晚两张科幻电影', '附近影院有什么', '50 元以内的 IMAX'].map((item) => (
            <button key={item} type="button" disabled={running} onClick={() => send(item)}>
              {item}
            </button>
          ))}
        </div>
      </section>

      <section className={styles.chat}>
        {messages.map((message) => (
          <div key={message.id} className={`${styles.message} ${message.role === 'user' ? styles.user : styles.assistant}`}>
            <div className={styles.bubble}>
              {message.status ? <span className={styles.loading}>{message.status}</span> : message.content}
            </div>
            <AgentCards cards={message.cards} disabled={running} onAction={send} />
          </div>
        ))}
      </section>

      <footer className={styles.composer}>
        <input
          value={input}
          disabled={running}
          placeholder="说出影片、时间、位置和人数"
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') void send(input);
          }}
        />
        <button type="button" disabled={running || !input.trim()} onClick={() => send(input)}>
          发送
        </button>
      </footer>
    </PrototypeShell>
  );
}
