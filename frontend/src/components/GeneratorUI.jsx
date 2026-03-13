import { useState, useTransition } from "react";


const DIGITS = Array.from({ length: 10 }, (_, value) => value);
const VARIATION_OPTIONS = [1, 3, 6];
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:5000").replace(/\/$/, "");

async function requestDigitImage(digit) {
  const response = await fetch(`${API_BASE_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      digit,
    }),
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || "Image generation failed.");
  }

  return payload;
}


function LoadingCard({ index }) {
  return (
    <div className="relative overflow-hidden rounded-[26px] border border-white/70 bg-white/80 p-5 shadow-panel">
      <div className="mb-4 flex items-center justify-between text-xs uppercase tracking-[0.28em] text-slate-600">
        <span>Rendering</span>
        <span>{String(index + 1).padStart(2, "0")}</span>
      </div>
      <div className="relative aspect-square overflow-hidden rounded-[20px] bg-slate-100">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/70 to-transparent animate-sweep" />
      </div>
    </div>
  );
}


export default function GeneratorUI() {
  const [digit, setDigit] = useState("0");
  const [variationCount, setVariationCount] = useState("3");
  const [images, setImages] = useState([]);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isPending, startTransition] = useTransition();

  async function handleGenerate() {
    setError("");
    setIsLoading(true);

    try {
      const totalVariations = Number(variationCount);
      const nextImages = await Promise.all(
        Array.from({ length: totalVariations }, async (_, index) => {
          const generated = await requestDigitImage(Number(digit));
          return {
            id: `${Date.now()}-${index}`,
            image: generated.image,
            digit: generated.digit,
          };
        }),
      );

      startTransition(() => {
        setImages(nextImages);
      });
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Image generation failed.");
    } finally {
      setIsLoading(false);
    }
  }

  const isBusy = isLoading || isPending;

  return (
    <main className="relative min-h-screen overflow-hidden px-4 py-6 sm:px-6 lg:px-10">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(235,185,88,0.35),_transparent_30%),radial-gradient(circle_at_85%_12%,_rgba(34,87,122,0.18),_transparent_22%),linear-gradient(135deg,_#f8f1e5_0%,_#efe6d3_42%,_#e3ecf4_100%)]" />

      <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-6 lg:flex-row">
        <section className="w-full rounded-[34px] border border-white/70 bg-white/75 p-6 shadow-panel backdrop-blur md:p-8 lg:max-w-md">
          <p className="mb-4 text-xs uppercase tracking-[0.4em] text-slate-700">AI Digit Studio</p>
          <h1 className="max-w-sm font-[Fraunces] text-4xl leading-tight text-ink sm:text-5xl">
            Generate fresh handwritten digits from your trained conditional GAN.
          </h1>
          <p className="mt-4 text-sm leading-7 text-slate-700 sm:text-base">
            Choose a class label, request one or many variations, and the app will sample new noise vectors to render new 28x28 digits in real time.
          </p>

          <div className="mt-8 space-y-5">
            <label className="block">
              <span className="mb-2 block text-sm font-medium uppercase tracking-[0.18em] text-slate-700">Digit label</span>
              <select
                value={digit}
                onChange={(event) => setDigit(event.target.value)}
                className="w-full rounded-2xl border border-slate-200 bg-sand px-4 py-3 text-lg font-semibold text-ink outline-none transition focus:border-brass"
              >
                {DIGITS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-medium uppercase tracking-[0.18em] text-slate-700">Variations</span>
              <select
                value={variationCount}
                onChange={(event) => setVariationCount(event.target.value)}
                className="w-full rounded-2xl border border-slate-200 bg-mist px-4 py-3 text-lg font-semibold text-ink outline-none transition focus:border-slate"
              >
                {VARIATION_OPTIONS.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>

            <button
              type="button"
              onClick={handleGenerate}
              disabled={isBusy}
              className="group inline-flex w-full items-center justify-center gap-3 rounded-2xl bg-ink px-5 py-4 text-sm font-semibold uppercase tracking-[0.24em] text-white transition hover:bg-slate disabled:cursor-not-allowed disabled:opacity-70"
            >
              <span className={isBusy ? "h-3 w-3 animate-pulse rounded-full bg-brass" : "h-3 w-3 rounded-full bg-brass transition group-hover:scale-125"} />
              {isBusy ? "Generating..." : "Generate images"}
            </button>

            <div className="rounded-[24px] border border-slate-200/70 bg-white/70 p-4 text-sm text-slate-700">
              Every request sends a new 100-dimensional noise vector to the Flask API, so repeated generations for the same label still produce distinct outputs.
            </div>

            {error ? <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div> : null}
          </div>
        </section>

        <section className="flex-1 rounded-[34px] border border-white/70 bg-[#102139]/90 p-6 shadow-panel backdrop-blur md:p-8">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-slate-300">Generated Output</p>
              <h2 className="mt-2 font-[Fraunces] text-3xl text-white sm:text-4xl">Digit {digit} variations</h2>
            </div>
            <p className="max-w-md text-sm leading-6 text-slate-300">
              The backend returns each generated image as a Base64 PNG, making the gallery simple to render without separate file storage.
            </p>
          </div>

          <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {isBusy
              ? Array.from({ length: Number(variationCount) }, (_, index) => <LoadingCard key={`loading-${index}`} index={index} />)
              : images.length > 0
                ? images.map((item, index) => (
                    <article key={item.id} className="rounded-[26px] border border-white/10 bg-white/10 p-5 text-white backdrop-blur">
                      <div className="mb-4 flex items-center justify-between text-xs uppercase tracking-[0.28em] text-slate-300">
                        <span>Variation</span>
                        <span>{String(index + 1).padStart(2, "0")}</span>
                      </div>
                      <div className="rounded-[20px] bg-white/95 p-4 shadow-inner">
                        <img
                          src={item.image}
                          alt={`Generated handwritten digit ${item.digit}`}
                          className="mx-auto aspect-square w-full max-w-[220px] rounded-xl border border-slate-200 bg-white object-contain p-2"
                        />
                      </div>
                    </article>
                  ))
                : (
                    <div className="col-span-full flex min-h-[320px] flex-col items-center justify-center rounded-[28px] border border-dashed border-white/20 bg-white/5 p-8 text-center text-slate-300">
                      <div className="mb-5 flex h-20 w-20 animate-float items-center justify-center rounded-full border border-white/15 bg-white/10 text-3xl font-semibold text-white">
                        {digit}
                      </div>
                      <h3 className="font-[Fraunces] text-2xl text-white">No images yet</h3>
                      <p className="mt-3 max-w-md text-sm leading-6 text-slate-300">
                        Pick a digit and generate a batch to preview how your conditional GAN interprets the selected class label.
                      </p>
                    </div>
                  )}
          </div>
        </section>
      </div>
    </main>
  );
}