/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        serif: ['"Newsreader"', 'Georgia', 'Cambria', '"Times New Roman"', 'serif'],
        sans: ['"Inter"', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'Consolas', 'monospace'],
      },
      colors: {
        canvas: '#F7F6F2',
        paper: '#FFFFFF',
        subtle: '#F1F0EC',
        hairline: '#E2E4E9',
        ink: {
          DEFAULT: '#0F172A',
          secondary: '#334155',
          muted: '#64748B',
        },
        crimson: {
          DEFAULT: '#A51C30',
          dark: '#821324',
          soft: '#FDF2F4',
          border: '#F2C6CE',
        },
        teal: {
          academic: '#0F766E',
          soft: '#F0FDFA',
          border: '#B2DFDB',
        },
        ochre: {
          DEFAULT: '#B45309',
          soft: '#FFFBEB',
          border: '#FDE68A',
        },
        navy: {
          DEFAULT: '#1E3A5F',
          soft: '#F0F5FA',
          border: '#CBD9E8',
        },
      },
      boxShadow: {
        card: '0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 1px rgba(15, 23, 42, 0.02)',
        elevated: '0 4px 16px rgba(15, 23, 42, 0.06), 0 1px 3px rgba(15, 23, 42, 0.04)',
      },
      animation: {
        'fade-in': 'fadeIn 0.25s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(3px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
};
