import { useEffect, useMemo, useState } from 'react';
import PrototypeShell from '@/components/PrototypeShell';
import SeatGrid from '@/components/SeatGrid';
import StepRail from '@/components/StepRail';
import {
  createOrder,
  getActiveDraft,
  getSeats,
  listMovies,
  listShowtimes,
  payOrder,
  updateDraft,
} from '@/services/api';
import { useAppStore } from '@/store/appStore';
import type { Movie, Order, Seat, Showtime } from '@/types';
import styles from './Home.module.css';

const fallbackMovies: Movie[] = [
  { movieId: 'm_1001', movieName: '流浪地球3', genre: '科幻', score: 9.1, durationMinutes: 150, status: 'NOW_SHOWING' },
  { movieId: 'm_1002', movieName: '喜剧之王', genre: '喜剧', score: 8.8, durationMinutes: 120, status: 'NOW_SHOWING' },
  { movieId: 'm_1003', movieName: '星际探险', genre: '科幻', score: 8.6, durationMinutes: 135, status: 'NOW_SHOWING' },
];

const fallbackShowtimes: Showtime[] = [
  { showtimeId: 'st_2001', movieId: 'm_1001', movieName: '流浪地球3', cinemaId: 'c_1001', cinemaName: 'Cinema One', hallName: '1号IMAX厅', hallType: 'IMAX', date: 'today', time: '19:30', price: 42, remainingSeats: 48 },
  { showtimeId: 'st_2002', movieId: 'm_1001', movieName: '流浪地球3', cinemaId: 'c_1002', cinemaName: 'Cinema Two', hallName: '2号激光厅', hallType: '普通', date: 'today', time: '21:10', price: 39, remainingSeats: 48 },
  { showtimeId: 'st_2004', movieId: 'm_1003', movieName: '星际探险', cinemaId: 'c_1002', cinemaName: 'Cinema Two', hallName: '1号IMAX厅', hallType: 'IMAX', date: 'today', time: '18:40', price: 45, remainingSeats: 48 },
];

function fallbackSeats(): Seat[] {
  return ['A', 'B', 'C', 'D', 'E', 'F'].flatMap((row) =>
    Array.from({ length: 8 }, (_, index) => ({
      seatId: `${row}${index + 1}`,
      row,
      number: index + 1,
      status: index === 1 && row === 'B' ? 'sold' : 'available',
      zone: row === 'C' || row === 'D' ? 'middle' : 'standard',
    })),
  );
}

export default function HomePage() {
  const { draft, setDraft, userId } = useAppStore();
  const [movies, setMovies] = useState<Movie[]>(fallbackMovies);
  const [showtimes, setShowtimes] = useState<Showtime[]>(fallbackShowtimes);
  const [seats, setSeats] = useState<Seat[]>([]);
  const [order, setOrder] = useState<Order | null>(null);

  useEffect(() => {
    getActiveDraft().then(setDraft).catch(() => undefined);
    listMovies().then((data) => setMovies(data.movies)).catch(() => undefined);
  }, [setDraft]);

  useEffect(() => {
    listShowtimes({ movieId: draft?.movieId || undefined, ticketCount: draft?.ticketCount || 2 })
      .then((data) => setShowtimes(data.showtimes))
      .catch(() => undefined);
  }, [draft?.movieId, draft?.ticketCount]);

  useEffect(() => {
    if (!draft?.showtimeId) {
      setSeats([]);
      return;
    }
    getSeats(draft.showtimeId).then((data) => setSeats(data.seats)).catch(() => setSeats(fallbackSeats()));
  }, [draft?.showtimeId]);

  const activeStep = useMemo(() => {
    if (order?.status === 'TICKETED') return 4;
    if (order) return 3;
    if (draft?.showtimeId) return 2;
    if (draft?.movieId) return 1;
    return 0;
  }, [draft?.movieId, draft?.showtimeId, order]);

  async function syncDraft(patch: Record<string, unknown>) {
    const next = await updateDraft({
      draftId: draft?.draftId,
      version: draft?.version,
      userId,
      ...patch,
    }).catch(() => undefined);
    if (next) setDraft(next);
  }

  function chooseMovie(movie: Movie) {
    void syncDraft({ movieId: movie.movieId, movieName: movie.movieName, ticketCount: draft?.ticketCount || 2 });
  }

  function chooseShowtime(showtime: Showtime) {
    void syncDraft({
      movieId: showtime.movieId,
      movieName: showtime.movieName,
      cinemaId: showtime.cinemaId,
      cinemaName: showtime.cinemaName,
      showtimeId: showtime.showtimeId,
      date: showtime.date,
      time: showtime.time,
    });
  }

  async function lockSeats(seatIds: string[]) {
    if (!draft?.showtimeId) return;
    const nextOrder = await createOrder({
      draftId: draft.draftId,
      showtimeId: draft.showtimeId,
      seatIds,
      ticketCount: draft.ticketCount,
      userId,
    }).catch(() => undefined);
    if (nextOrder) setOrder(nextOrder);
  }

  async function simulatePay() {
    if (!order) return;
    const paid = await payOrder(order.orderId, crypto.randomUUID()).catch(() => undefined);
    if (paid) setOrder(paid);
  }

  return (
    <PrototypeShell mode="traditional">
      <StepRail active={activeStep} />
      <section className={styles.hero}>
        <div>
          <span className={styles.badge}>共享草稿</span>
          <h1>今晚看什么？</h1>
          <p>{draft?.movieName || '先选影片'} · {draft?.cinemaName || '再选影院'} · {draft?.ticketCount || 2} 张</p>
        </div>
        <button type="button" onClick={() => { window.location.hash = `/agent?draftId=${draft?.draftId || ''}`; }}>
          交给 AI
        </button>
      </section>

      <section className={styles.section}>
        <div className={styles.titleRow}>
          <h2>热映影片</h2>
          <span>影片频道</span>
        </div>
        <div className={styles.movieRail}>
          {movies.map((movie) => (
            <button
              key={movie.movieId}
              type="button"
              className={`${styles.movieCard} ${draft?.movieId === movie.movieId ? styles.selected : ''}`}
              onClick={() => chooseMovie(movie)}
            >
              <div className={styles.poster}>{movie.genre}</div>
              <strong>{movie.movieName}</strong>
              <span>{movie.score} 分 · {movie.durationMinutes} 分钟</span>
            </button>
          ))}
        </div>
      </section>

      <section className={styles.section}>
        <div className={styles.titleRow}>
          <h2>推荐场次</h2>
          <span>余座 / 价格 / 厅型</span>
        </div>
        <div className={styles.showtimeList}>
          {showtimes.map((showtime) => (
            <button
              key={showtime.showtimeId}
              type="button"
              className={`${styles.showtime} ${draft?.showtimeId === showtime.showtimeId ? styles.selected : ''}`}
              onClick={() => chooseShowtime(showtime)}
            >
              <strong>{showtime.time}</strong>
              <span>{showtime.movieName} · {showtime.cinemaName}</span>
              <em>{showtime.hallType} · ￥{showtime.price} · 余座 {showtime.remainingSeats ?? '-'}</em>
            </button>
          ))}
        </div>
      </section>

      {draft?.showtimeId ? (
        <SeatGrid seats={seats.length ? seats : fallbackSeats()} limit={draft.ticketCount || 2} onConfirm={lockSeats} />
      ) : null}

      {order ? (
        <section className={styles.ticketPanel}>
          <div className={styles.titleRow}>
            <h2>{order.status === 'TICKETED' ? '电子票' : '订单确认'}</h2>
            <span>{order.status}</span>
          </div>
          <p>{order.movieName} · {order.cinemaName}</p>
          <p>{order.date} {order.time} · {order.hallName}</p>
          <p>座位 {order.seatIds.join('、')} · 总额 ￥{order.amount}</p>
          {order.ticketCodes?.length ? (
            <div className={styles.codeBox}>{order.ticketCodes.join(' / ')}</div>
          ) : (
            <button type="button" onClick={simulatePay}>模拟支付成功</button>
          )}
        </section>
      ) : null}
    </PrototypeShell>
  );
}
