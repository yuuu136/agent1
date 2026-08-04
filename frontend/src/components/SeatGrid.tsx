import { useState } from 'react';
import type { Seat } from '@/types';
import styles from './SeatGrid.module.css';

interface SeatGridProps {
  seats: Seat[];
  limit: number;
  onConfirm: (seatIds: string[]) => void;
}

export default function SeatGrid({ seats, limit, onConfirm }: SeatGridProps) {
  const [selected, setSelected] = useState<string[]>([]);

  function toggle(seat: Seat) {
    if (seat.status !== 'available') return;
    setSelected((current) => {
      if (current.includes(seat.seatId)) return current.filter((item) => item !== seat.seatId);
      if (current.length >= limit) return current;
      return [...current, seat.seatId];
    });
  }

  return (
    <section className={styles.card}>
      <div className={styles.title}>
        <strong>共享座位图</strong>
        <span>{selected.length}/{limit}</span>
      </div>
      <div className={styles.screen}>银幕</div>
      <div className={styles.grid}>
        {seats.map((seat) => (
          <button
            key={seat.seatId}
            type="button"
            className={[
              styles.seat,
              styles[`seat_${seat.status}`] || '',
              selected.includes(seat.seatId) ? styles.selected : '',
            ].join(' ')}
            disabled={seat.status !== 'available'}
            onClick={() => toggle(seat)}
          >
            {seat.seatId}
          </button>
        ))}
      </div>
      <div className={styles.legend}>
        <span>可选</span>
        <span>已选</span>
        <span>锁定/已售</span>
      </div>
      <button
        className={styles.confirm}
        type="button"
        disabled={!selected.length}
        onClick={() => onConfirm(selected)}
      >
        确认座位并锁座
      </button>
    </section>
  );
}
