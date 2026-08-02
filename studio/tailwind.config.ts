import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        night: {
          950: "#070912",
          900: "#0b0e1d",
          800: "#12162e",
          700: "#1a1f40",
          600: "#262c58",
          500: "#39406e",
        },
        gold: {
          200: "#f6ecc8",
          300: "#efdda2",
          400: "#e4c778",
          500: "#d4ab4a",
          600: "#b98f33",
          700: "#93702a",
        },
        emerald: {
          400: "#34d399",
          500: "#10b981",
        },
        amber: {
          400: "#fbbf24",
          500: "#f59e0b",
        },
        rose: {
          400: "#f472b6",
          500: "#f43f5e",
        },
      },
      fontFamily: {
        display: ["Georgia", "Times New Roman", "serif"],
        sans: ["ui-sans-serif", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Helvetica Neue", "sans-serif"],
        mono: ["ui-monospace", "SF Mono", "Monaco", "Inconsolata", "Fira Mono", "Droid Sans Mono", "monospace"],
      },
      boxShadow: {
        glow: "0 0 24px rgba(212, 171, 74, 0.12)",
      },
      animation: {
        "fade-up": "fade-up 0.25s ease-out both",
        "pulse-gold": "pulse-gold 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
      keyframes: {
        "fade-up": {
          "from": {
            opacity: "0",
            transform: "translateY(6px)",
          },
          "to": {
            opacity: "1",
            transform: "translateY(0)",
          },
        },
        "pulse-gold": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
