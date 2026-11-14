"use strict";

const byId = (id) => document.getElementById(id);
const metrics = FieldLab.metrics;
const format = new Intl.NumberFormat("en", {maximumFractionDigits: 2});
const number = (value) => value == null ? "—" : format.format(value);
const mean = FieldLab.mean;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function options(select, entries, initial) {
  select.replaceChildren(...entries.map(([value, label]) => new Option(label, value)));
  select.value = initial;
}

function drawChart(canvas, series, repeats, key, selectedRepeat) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const values = series.flatMap(({rows}) => rows.map((row) => row[key]));
  const low = Math.min(...values);
  const high = Math.max(...values);
  const pad = (high - low) * .1 || Math.max(1, Math.abs(low) * .05);
  const min = low - pad;
  const max = high + pad;
  const left = 59, right = width - 17, top = 18, bottom = height - 38;
  const x = (index) => repeats.length === 1 ? (left + right) / 2 :
    left + index * (right - left) / (repeats.length - 1);
  const y = (value) => bottom - (value - min) * (bottom - top) / (max - min);
  const positions = new Map(repeats.map((repeat, index) => [repeat, index]));
  context.font = "11px system-ui";
  context.fillStyle = "#52665d";
  context.textAlign = "right";
  for (let step = 0; step <= 4; step++) {
    const value = min + (max - min) * step / 4;
    const row = y(value);
    context.strokeStyle = "#d7dfd4";
    context.beginPath();
    context.moveTo(left, row);
    context.lineTo(right, row);
    context.stroke();
    context.fillText(new Intl.NumberFormat("en", {notation: "compact", maximumFractionDigits: 1}).format(value), left - 7, row + 4);
  }

  context.textAlign = "center";
  const stride = Math.max(1, Math.ceil(repeats.length / Math.max(2, Math.floor((right - left) / 29))));
  repeats.forEach((repeat, index) => {
    if (index % stride === 0) context.fillText(repeat, x(index), bottom + 16);
  });
  context.fillText("Archived repeat", (left + right) / 2, height - 5);

  for (const {rows, color, label} of series) {
    context.fillStyle = color;
    context.strokeStyle = color;
    context.lineWidth = 2;
    for (const row of rows) {
      const px = x(positions.get(row.repeat));
      const py = y(row[key]);
      context.globalAlpha = selectedRepeat && row.repeat !== selectedRepeat ? .3 : 1;
      if (label === "A") {
        context.beginPath();
        context.arc(px, py, 4, 0, Math.PI * 2);
        context.fill();
      } else {
        context.strokeRect(px - 4, py - 4, 8, 8);
      }
    }
  }
  context.globalAlpha = 1;
}

function start(data) {
  const runs = new Map(data.map((run) => [run.id, run.rows]));
  const ids = [...runs.keys()].sort((a, b) => a.localeCompare(b, "en", {numeric: true}));
  const first = "model_ep949", second = "model_ep953";
  const defaults = {a: runs.has(first) ? first : ids[0], b: runs.has(second) ? second : ids.at(-1), metric: "reward"};
  const controls = Object.fromEntries(Object.keys(defaults).map((key) => [key, byId(key)]));
  const query = new URLSearchParams(location.search);
  const checkpoints = ids.map((id) => [id, id]);
  for (const key of ["a", "b"]) {
    const candidate = query.get(key);
    options(controls[key], checkpoints, runs.has(candidate) ? candidate : defaults[key]);
  }
  const candidate = query.get("metric");
  options(controls.metric, Object.entries(metrics).map(([key, [label]]) => [key, label]),
    Object.hasOwn(metrics, candidate) ? candidate : defaults.metric);
  byId("dataset").textContent = `${format.format(data.length)} checkpoints · ${format.format(data.reduce((n, run) => n + run.rows.length, 0))} evaluations`;

  const canvas = element("canvas");
  canvas.setAttribute("role", "img");
  byId("chart").append(canvas);
  const read = () => Object.fromEntries(Object.entries(controls).map(([key, input]) => [key, input.value]));
  const comparison = ({a, b}) => [
    {id: a, rows: runs.get(a), label: "A", color: "#21654d"},
    {id: b, rows: runs.get(b), label: "B", color: "#a64b27"},
  ];
  const repeatIds = (series) => [...new Set(series.flatMap(({rows}) => rows.map(({repeat}) => repeat)))].sort((a, b) => a - b);
  const redraw = () => {
    const state = read();
    const series = comparison(state);
    drawChart(canvas, series, repeatIds(series), state.metric, Number(byId("repeat").value));
  };

  function render() {
    byId("status").textContent = "";
    const state = read();
    const series = comparison(state);
    const [label, unit] = metrics[state.metric];
    if (location.protocol !== "file:") {
      const url = new URL(location.href);
      for (const [key, value] of Object.entries(state)) url.searchParams.set(key, value);
      try {
        history.replaceState(null, "", url);
      } catch {
        byId("status").textContent = "The browser could not save the selection in the address. Comparison and export remain available.";
      }
    }

    byId("comparison-note").textContent = state.a === state.b ?
      "Both selections refer to the same checkpoint. Choose another checkpoint to compare policies." :
      `A · ${state.a} compared with B · ${state.b}`;
    byId("metric-stats").replaceChildren(...series.map(({id, label: side, rows}) =>
      element("p", `${side} · ${id} — mean ${number(mean(rows, state.metric))} ${unit}; range ${number(Math.min(...rows.map((row) => row[state.metric])))}–${number(Math.max(...rows.map((row) => row[state.metric])))} ${unit}`)));
    byId("summary").replaceChildren(...["yield", "irr", "fert"].map((key) => {
      const [name, units] = metrics[key];
      const averages = series.map(({rows}) => mean(rows, key));
      const delta = averages[1] - averages[0];
      const card = element("article", undefined, "card");
      const list = element("dl");
      ["A mean", "B mean", "B − A"].forEach((text, index) => {
        const value = [...averages, delta][index];
        list.append(element("dt", text), element("dd", `${index === 2 && value > 0 ? "+" : ""}${number(value)} ${units}`));
      });
      card.append(element("h3", name), list);
      return card;
    }));
    byId("chartTitle").textContent = `${label} (${unit})`;
    canvas.setAttribute("aria-label", `${label} by archived repeat. Exact values are available in the adjacent table.`);
    byId("legend").replaceChildren(...series.map(({id, label, rows}) =>
      element("span", `${label === "A" ? "●" : "□"} ${label} · ${id} · ${rows.length} repeats`, `series-${label.toLowerCase()}`)));
    const heading = element("tr");
    for (const text of ["Repeat", "A", "B"]) {
      const cell = element("th", text);
      cell.scope = "col";
      heading.append(cell);
    }
    byId("columns").replaceChildren(heading);
    const rowMaps = series.map(({rows}) => new Map(rows.map((row) => [row.repeat, row[state.metric]])));
    byId("observations").replaceChildren(...repeatIds(series).map((repeat) => {
      const row = element("tr");
      const key = element("th", repeat);
      key.scope = "row";
      row.append(key, ...rowMaps.map((map) => element("td", number(map.get(repeat)))));
      return row;
    }));

    const repeats = repeatIds(series);
    const selected = byId("repeat").value;
    options(byId("repeat"), [["", "All repeats"], ...repeats.map((repeat) => [String(repeat), String(repeat)])],
      repeats.includes(Number(selected)) ? selected : "");
    filterCheckpoints();
    inspectRepeat();
  }

  function inspectRepeat() {
    const state = read();
    const repeat = Number(byId("repeat").value);
    const [label, unit] = metrics[state.metric];
    byId("point-detail").textContent = repeat ? comparison(state).map(({label: side, rows}) =>
      `${side} · repeat ${repeat}: ${number(rows.find((row) => row.repeat === repeat)?.[state.metric])} ${unit}`).join(" · ") :
      `${label}: all archived repeats`;
    for (const row of byId("observations").children) {
      row.classList.toggle("selected-repeat", Number(row.firstChild.textContent) === repeat);
    }
    redraw();
  }
  byId("repeat").addEventListener("change", inspectRepeat);

  function filterCheckpoints() {
    const query = byId("search").value.trim().toLowerCase();
    const matches = checkpoints.filter(([id]) => id.toLowerCase().includes(query));
    for (const key of ["a", "b"]) {
      const current = controls[key].value;
      const entries = matches.some(([id]) => id === current) ? matches : [[current, current], ...matches];
      options(controls[key], entries, current);
    }
    byId("search-result").textContent = query ?
      `${matches.length} matching checkpoints. Current selections stay available.` : "";
  }
  byId("search").addEventListener("input", filterCheckpoints);
  for (const select of Object.values(controls)) select.addEventListener("change", render);
  byId("swap").addEventListener("click", () => {
    const state = read();
    byId("search").value = "";
    byId("search-result").textContent = "";
    options(controls.a, checkpoints, state.b);
    options(controls.b, checkpoints, state.a);
    render();
  });
  byId("reset").addEventListener("click", () => {
    byId("search").value = "";
    byId("search-result").textContent = "";
    for (const key of ["a", "b"]) options(controls[key], checkpoints, defaults[key]);
    controls.metric.value = defaults.metric;
    byId("repeat").value = "";
    render();
  });
  byId("export").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([FieldLab.toCsv(comparison(read()))], {type:"text/csv;charset=utf-8"}));
    const link = element("a");
    link.href = url;
    const state = read();
    link.download = `fieldlab-${state.a}-vs-${state.b}.csv`;
    document.body.append(link);
    link.click();
    link.remove();
    byId("status").textContent = "CSV download requested. All six metrics and both selected checkpoints are included.";
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  byId("workspace").hidden = false;
  render();
  new ResizeObserver(redraw).observe(byId("chart"));
}

try {
  FieldLab.validate(window.FIELDLAB_DATA);
  start(window.FIELDLAB_DATA);
} catch (error) {
  byId("workspace").hidden = true;
  byId("error").textContent = "The explorer could not load. Check that data.js is beside index.html, or use the repository’s Python results demo.";
  byId("recovery").hidden = false;
  console.error(error);
}
