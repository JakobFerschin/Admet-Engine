/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', '"Cascadia Code"', 'Consolas', 'monospace'],
        sans: ['"Inter"', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        surface: {
          DEFAULT: '#0a0a0f',   // bg-gray-950 level
          raised:   '#111118',   // cards / panels
          overlay:  '#1a1a24',   // inputs / interactive
          border:   '#2a2a3a',   // subtle borders
        },
        accent: {
          cyan:    '#22d3ee',    // data highlights
          neon:    '#00f0ff',    // peak values
          amber:   '#f59e0b',    // warnings (yellow)
          rose:    '#f43f5e',    // danger (red)
          emerald: '#10b981',    // safe (green)
          violet:  '#a78bfa',    // secondary accent
        },
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
