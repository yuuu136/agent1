import { defineConfig } from '@umijs/max';

export default defineConfig({
  title: '光影票务',
  npmClient: 'npm',
  history: { type: 'hash' },
  esbuildMinifyIIFE: true,
  routes: [
    { path: '/', redirect: '/home' },
    { path: '/home', component: './Home' },
    { path: '/agent', component: './Agent' },
  ],
  proxy: {
    '/api': {
      target: 'http://127.0.0.1:8001',
      changeOrigin: true,
    },
    '/agent': {
      target: 'http://127.0.0.1:8001',
      changeOrigin: true,
    },
  },
});
