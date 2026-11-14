const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const model = require("../docs/fieldlab/model.js");
const context = {window: {}};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, "../docs/fieldlab/data.js"), "utf8"), context);
const data = JSON.parse(JSON.stringify(context.window.FIELDLAB_DATA));
test("archive counts and mean match the source", () => {
  model.validate(data);
  assert.equal(data.length, 129);
  assert.equal(data.flatMap(run=>run.rows).length, 1290);
  assert.equal(model.mean(data.find(run=>run.id==="model_ep949").rows, "irr"), 275.4);
});
test("reject invalid bundles before rendering", () => {
  for (const invalid of [null, [], {}, [{id:"bad", rows:[]}], [data[0], data[0]]]) {
    assert.throws(()=>model.validate(invalid));
  }
  for (const value of [NaN, Infinity, null, "12", -1]) {
    const copy = JSON.parse(JSON.stringify(data.slice(0,1)));
    copy[0].rows[0].irr = value;
    assert.throws(()=>model.validate(copy));
  }
  const copy = JSON.parse(JSON.stringify(data.slice(0,1)));
  copy[0].rows.push(copy[0].rows[0]);
  assert.throws(()=>model.validate(copy));
});
test("CSV includes both sides, six metrics, and unrounded numbers", () => {
  const series = data.slice(0,2).map((run,i)=>({...run,label:i?"B":"A"}));
  const csv = model.toCsv(series);
  assert.equal(csv.trimEnd().split("\r\n").length, 21);
  assert.ok(csv.includes('"1298.1515546875"'));
  assert.ok(csv.includes('"B","model_ep953"'));
  assert.ok(csv.startsWith('"side","checkpoint","repeat","reward","yield","irr","fert","wue","nue"'));
});
