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
      },
      fontFamily: {
        serif: ["Georgia", "serif"],
        mono: ["Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
