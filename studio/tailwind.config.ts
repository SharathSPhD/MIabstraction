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
        paper: "#FBFAF7",
        panel: "#F2F1EC",
        ink: "#1F232B",
        body: "#373C46",
        muted: "#6B6F79",
        accent: "#3B4CC0",
        verified: "#2E7D4F",
        refusal: "#D97706",
      },
      fontFamily: {
        serif: ["Source Serif 4", "Georgia", "serif"],
        sans: ["ui-sans-serif", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Helvetica Neue", "sans-serif"],
        mono: ["ui-monospace", "SF Mono", "Monaco", "Inconsolata", "Fira Mono", "Droid Sans Mono", "monospace"],
      },
      fontSize: {
        "display": ["48px", { lineHeight: "1.1", fontWeight: "700", letterSpacing: "-0.02em" }],
        "subheading": ["28px", { lineHeight: "1.2", fontWeight: "600" }],
      },
      letterSpacing: {
        "tight": "-0.01em",
      },
      borderWidth: {
        "hairline": "1px",
      },
      underlineOffset: {
        "3": "3px",
      },
      boxShadow: {
        sm: "0 1px 2px 0 rgba(31, 35, 43, 0.06)",
        md: "0 4px 12px 0 rgba(31, 35, 43, 0.1)",
        lg: "0 10px 24px 0 rgba(31, 35, 43, 0.12)",
        "accent-glow": "0 0 20px rgba(59, 76, 192, 0.15)",
        "accent-sm": "0 0 12px rgba(59, 76, 192, 0.08)",
      },
      spacing: {
        "4.5": "1.125rem",
        "13": "3.25rem",
        "14": "3.5rem",
        "15": "3.75rem",
      },
      animation: {
        "pulse-subtle": "pulse-subtle 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-up": "fade-up 0.25s ease-out both",
      },
      keyframes: {
        "pulse-subtle": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.7" },
        },
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
      },
      maxWidth: {
        "prose": "65ch",
        "screen-xl": "1400px",
      },
    },
  },
  plugins: [],
};

export default config;
