"use strict";

/** Data validation and statistics shared by rendering and regression checks. */
const FieldLab = {
  metrics: {
    reward: ["Reward", "score"],
    yield: ["Harvest yield", "kg/ha"],
    irr: ["Irrigation", "mm"],
    fert: ["Nitrogen", "kg/ha"],
    wue: ["Water use efficiency", "kg/ha/mm"],
    nue: ["Nitrogen use efficiency", "kg/kg"],
  },

  validate(data) {
    if (!Array.isArray(data) || data.length === 0) throw new Error("Empty archive");
    const ids = new Set();
    for (const run of data) {
      if (!run || !/^model_ep[0-9]+$/.test(run.id) || ids.has(run.id)) throw new Error("Invalid or duplicate checkpoint");
      ids.add(run.id);
      if (!Array.isArray(run.rows) || run.rows.length === 0) throw new Error("Empty checkpoint");
      const repeats = new Set();
      for (const row of run.rows) {
        if (!row || !Number.isSafeInteger(row.repeat) || row.repeat < 1 || repeats.has(row.repeat)) {
          throw new Error("Invalid or duplicate repeat");
        }
        repeats.add(row.repeat);
        for (const key of Object.keys(this.metrics)) {
          if (!Number.isFinite(row[key]) || (key !== "reward" && row[key] < 0)) throw new Error("Invalid metric");
        }
      }
    }
    return data;
  },

  mean(rows, key) {
    return rows.reduce((sum, row) => sum + row[key], 0) / rows.length;
  },

  toCsv(series) {
    const keys = Object.keys(this.metrics);
    const cell = (value) => '"' + String(value).replaceAll('"', '""') + '"';
    const lines = [["side", "checkpoint", "repeat", ...keys], ...series.flatMap(({id, label, rows}) =>
      rows.map((row) => [label, id, row.repeat, ...keys.map((key) => row[key])]))];
    return lines.map((line) => line.map(cell).join(",")).join("\r\n") + "\r\n";
  },
};
if (typeof module !== "undefined") module.exports = FieldLab;
