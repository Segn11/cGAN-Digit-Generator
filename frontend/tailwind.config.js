/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#171414",
        sand: "#efe6d3",
        brass: "#b98737",
        slate: "#20304f",
        mist: "#d9e7f2",
      },
      boxShadow: {
        panel: "0 28px 60px rgba(31, 24, 16, 0.12)",
      },
      animation: {
        float: "float 5s ease-in-out infinite",
        sweep: "sweep 1.2s ease-in-out infinite",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-10px)" },
        },
        sweep: {
          "0%": { transform: "translateX(-120%)" },
          "100%": { transform: "translateX(120%)" },
        },
      },
    },
  },
  plugins: [],
};