export interface ApiResponse<T> {
  code: number;
  message: string;
  data: T;
  traceId: string;
}

export interface Movie {
  movieId: string;
  movieName: string;
  genre: string;
  score: number;
  durationMinutes: number;
  status: string;
}

export interface Showtime {
  showtimeId: string;
  movieId: string;
  movieName: string;
  cinemaId: string;
  cinemaName: string;
  hallName: string;
  hallType: string;
  date: string;
  time: string;
  price: number;
  remainingSeats?: number;
}

export interface Seat {
  seatId: string;
  row: string;
  number: number;
  status: string;
  zone?: string;
}

export interface PurchaseDraft {
  draftId: number;
  version: number;
  userId: string;
  state: string;
  movieId?: string | null;
  movieName?: string | null;
  cinemaId?: string | null;
  cinemaName?: string | null;
  showtimeId?: string | null;
  date?: string | null;
  time?: string | null;
  ticketCount: number;
  seatIds: string[];
  orderId?: string | null;
}

export interface Order {
  orderId: string;
  showtimeId: string;
  seatIds: string[];
  movieName: string;
  cinemaName: string;
  hallName: string;
  date: string;
  time: string;
  amount: number;
  status: string;
  expiresAt?: string;
  ticketStatus?: string;
  ticketCodes?: string[];
}

export interface AgentCard {
  type: string;
  id?: string;
  title?: string;
  subtitle?: string;
  meta?: Record<string, unknown>;
  payload?: Record<string, unknown>;
  seats?: Seat[];
  actions?: Array<{ event: string; label: string; payload?: Record<string, unknown> }>;
}

export interface AgentMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  status?: string;
  cards?: AgentCard[];
}
