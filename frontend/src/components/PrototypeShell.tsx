import type { ReactNode } from 'react';
import styles from './PrototypeShell.module.css';

interface PrototypeShellProps {
  mode: 'traditional' | 'agent';
  children: ReactNode;
}

export default function PrototypeShell({ mode, children }: PrototypeShellProps) {
  return (
    <main className={styles.canvas}>
      <section className={styles.phone}>
        <header className={styles.top}>
          <div>
            <span className={styles.city}>洛阳</span>
            <span className={styles.locating}>定位中</span>
          </div>
          <button className={styles.iconButton} type="button" aria-label="我的">
            个人
          </button>
        </header>
        <div className={styles.search}>搜索影片、影院、商圈</div>
        <nav className={styles.modeSwitch}>
          <button
            type="button"
            className={mode === 'traditional' ? styles.active : ''}
            onClick={() => { window.location.hash = '/home'; }}
          >
            传统购票
          </button>
          <button
            type="button"
            className={mode === 'agent' ? styles.active : ''}
            onClick={() => { window.location.hash = '/agent'; }}
          >
            AI 购票
          </button>
        </nav>
        {children}
        <footer className={styles.bottomNav}>
          <span className={styles.bottomActive}>首页</span>
          <span>订单</span>
          <span>我的</span>
        </footer>
      </section>
    </main>
  );
}
