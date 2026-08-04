import styles from './StepRail.module.css';

const steps = ['需求', '场次', '选座', '支付', '出票'];

export default function StepRail({ active = 0 }: { active?: number }) {
  return (
    <div className={styles.rail}>
      {steps.map((step, index) => (
        <div key={step} className={index <= active ? styles.done : ''}>
          <span>{index + 1}</span>
          <strong>{step}</strong>
        </div>
      ))}
    </div>
  );
}
