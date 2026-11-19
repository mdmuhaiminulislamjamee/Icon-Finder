// State and interaction regressions for the editor, without an external browser.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "dashboard/static/app.js"), "utf8");

function editor() {
  const nodes = new Map();
  const drawing = new Proxy({measureText: () => ({width: 100})}, {
    get: (target, key) => key in target ? target[key] : () => {},
  });
  function element(id) {
    if (!nodes.has(id)) nodes.set(id, {
      value: id.startsWith("filter-") && id !== "filter-search" ? "all" : "",
      textContent: "", innerHTML: "", disabled: false, options: [], width: 100, height: 200,
      clientWidth: 100, clientHeight: 200,
      classList: {add() {}, remove() {}, toggle() {}}, style: {},
      addEventListener() {}, getContext: () => drawing,
      getBoundingClientRect: () => ({left: 0, top: 0, width: 100, height: 200}),
      add(option) {this.options.push(option);}, replaceChildren(...items) {this.options = items;},
    });
    return nodes.get(id);
  }
  const alerts = [];
  const context = vm.createContext({
    console, document: {addEventListener() {}, getElementById: element},
    window: {addEventListener() {}},
    Image: class {
      constructor() {this.complete = true; this.naturalWidth = 100; this.naturalHeight = 200;}
      set src(value) {this.url = value; this.onload?.();}
    },
    Option: class {constructor(text, value) {this.text = text; this.value = value;}},
    setTimeout: (callback) => {callback(); return 1;}, clearTimeout() {},
    setInterval() {}, clearInterval() {},
    alert: (message) => alerts.push(message), confirm: () => true,
    fetch: async () => ({ok: true, json: async () => []}),
  });
  const run = (code) => vm.runInContext(code, context);
  vm.runInContext(source, context);
  run(`initCanvas(); allRecords = [{record_id:"sample", platform:"youtube", device:"tablet", content_format:"short_video", orientation:"portrait", curation_status:"raw", raw_annotations:[], effective_annotations:[{class:"save",xyxy:[10,20,40,60]},{class:"save",xyxy:[10,20,40,60]}], display_annotations:[{class:"save",xyxy:[10,20,40,60],source:"ios_xctest_accessibility"}], annotation_quality:{duplicates_removed:1}, image_url:"/sample.png"}]; openEditor("sample");`);
  element("draw-class-select").value = "save";
  return {context, run, element, alerts};
}

async function main() {
  let passed = 0;
  async function check(name, callback) {
    await callback();
    passed++;
    console.log(`PASS ${name}`);
  }
  await check("raw captures open in needs_review with a deduplicated display", () => {
    const e = editor();
    assert.equal(e.element("curation-status-select").value, "needs_review");
    assert.equal(e.run("currentAnnotations.length"), 1);
    assert.match(e.element("annotation-quality-message").textContent, /1 duplicate/);
  });
  await check("drawing ignores duplicate boxes and resets approved status", () => {
    const e = editor();
    e.element("curation-status-select").value = "approved";
    e.run("onMouseDown({button:0,clientX:60,clientY:20}); onMouseUp({clientX:90,clientY:60});");
    assert.equal(e.run("currentAnnotations.length"), 2);
    assert.equal(e.run("Object.hasOwn(currentAnnotations[1], 'interaction_xyxy')"), false);
    assert.equal(e.element("curation-status-select").value, "needs_review");
    e.run("onMouseDown({button:0,clientX:60,clientY:20}); onMouseUp({clientX:90,clientY:60});");
    assert.equal(e.run("currentAnnotations.length"), 2);
  });
  await check("an existing box can be relabelled and resized", () => {
    const e = editor();
    e.run("editAnnotation(0)");
    e.element("draw-class-select").value = "share";
    e.element("box-x1").value = "15";
    e.element("box-x2").value = "45";
    e.run("applyBoxChanges()");
    assert.equal(e.run("currentAnnotations[0].class"), "share");
    assert.equal(e.run("currentAnnotations[0].xyxy.join(',')"), "15,20,45,60");
  });
  await check("accepted suggestions become reviewed boxes and rejections stay out", () => {
    const e = editor();
    e.run(`currentProposals=[{class:"like",xyxy:[60,70,90,100],proposal_id:"yes",review_required:true},{class:"pause",xyxy:[40,110,70,150],proposal_id:"no",review_required:true}]; proposalDecisions=currentProposals.map(p=>({...p,decision:"pending"})); reviewSuggestion(0,true); reviewSuggestion(0,false);`);
    assert.equal(e.run("currentAnnotations.length"), 2);
    assert.equal(e.run("currentAnnotations[1].review_required"), false);
    assert.equal(e.run("proposalDecisions.map(p=>p.decision).join(',')"), "accepted,rejected");
    assert.equal(e.run("currentProposals.length"), 0);
  });
  await check("pending saved suggestions reopen outside the annotation set", () => {
    const e = editor();
    e.run(`allRecords[0].proposal_review=[{class:"share",xyxy:[60,70,90,100],proposal_id:"pending",decision:"pending",review_required:true}]; openEditor("sample");`);
    assert.equal(e.run("currentProposals.length"), 1);
    assert.equal(e.run("currentAnnotations.length"), 1);
    assert.equal(e.run("currentProposals[0].class"), "share");
  });
  await check("native capture boxes require review and reopen from saved pending decisions", () => {
    const e = editor();
    e.run(`allRecords[0].display_annotations[0].review_required=true; openEditor("sample");`);
    assert.equal(e.run("currentAnnotations.length"), 0);
    assert.equal(e.run("currentProposals.length"), 1);
    assert.equal(e.run("currentProposals[0].proposal_kind"), "native_capture");
    e.run(`allRecords[0].display_annotations=[]; allRecords[0].proposal_review=proposalDecisions; openEditor("sample");`);
    assert.equal(e.run("currentProposals.length"), 1);
    assert.equal(e.run("currentAnnotations.length"), 0);
    e.run("reviewSuggestion(0,true)");
    assert.equal(e.run("currentAnnotations[0].review_required"), false);
    assert.equal(e.run("currentAnnotations[0].geometry_reviewed"), true);
  });
  await check("rejected native candidates stay rejected after reopening", () => {
    const e = editor();
    e.run(`allRecords[0].display_annotations[0].review_required=true; openEditor("sample"); reviewSuggestion(0,false); allRecords[0].proposal_review=proposalDecisions; openEditor("sample");`);
    assert.equal(e.run("currentAnnotations.length"), 0);
    assert.equal(e.run("currentProposals.length"), 0);
  });
  await check("opening a capture automatically recovers missing boxes without manual suggestion", async () => {
    const e = editor();
    const urls = [];
    e.context.fetch = async (url) => {
      urls.push(url);
      return {ok: true, json: async () => ({proposals:[{class:"volume_muted",xyxy:[60,70,90,100],proposal_id:"muted-auto",review_required:true}],warnings:[]})};
    };
    e.run(`allRecords[0].has_accessibility_snapshot=true; openEditor("sample");`);
    await new Promise(setImmediate);
    assert.deepEqual(urls, ["/api/detect/youtube/tablet/short_video/sample"]);
    assert.equal(e.run("currentProposals[0].class"), "volume_muted");
    assert.equal(e.run("currentAnnotations.length"), 1);
    assert.equal(e.run("proposalDecisions[0].decision"), "pending");
    assert.match(e.element("suggestion-message").textContent, /Recovered 1 additional/);
  });
  await check("automatic recovery preserves pending work and saved rejections", async () => {
    const e = editor();
    e.run(`allRecords[0].proposal_review=[{class:"like",xyxy:[60,70,90,100],proposal_id:"no",decision:"rejected"},{class:"share",xyxy:[60,110,90,140],proposal_id:"keep",decision:"pending"}];`);
    e.context.fetch = async () => ({ok:true,json:async()=>({proposals:[{class:"like",xyxy:[60,70,90,100],proposal_id:"no-again"}],warnings:[]})});
    e.run(`allRecords[0].has_accessibility_snapshot=true; openEditor("sample");`);
    await new Promise(setImmediate);
    assert.equal(e.run("currentProposals.map(p=>p.class).join(',')"), "share");
    assert.equal(e.run("proposalDecisions.map(p=>p.decision).join(',')"), "rejected,pending");
  });
  await check("automatic correction replaces an outdated pending box and Zoom isolates the new one", async () => {
    const e = editor();
    e.context.fetch = async () => ({ok:true,json:async()=>({proposals:[{
      class:"play",xyxy:[45,75,81,120],proposal_id:"corrected-play",action:"replace_geometry",
      replaces:{class:"play",xyxy:[29,54,92,109]},review_required:true
    }],warnings:[]})});
    e.run(`allRecords[0].display_annotations=[{class:"play",xyxy:[29,54,92,109],review_required:true}];
      allRecords[0].has_accessibility_snapshot=true; openEditor("sample");`);
    await new Promise(setImmediate);
    assert.equal(e.run("currentProposals.length"), 1);
    assert.equal(e.run("currentProposals[0].xyxy.join(',')"), "45,75,81,120");
    assert.equal(e.run("proposalDecisions.map(p=>p.decision).join(',')"), "superseded,pending");
    assert.equal(e.run("currentAnnotations.length"), 0);
    e.run("zoomToBox(0,true)");
    assert.equal(e.element("annotation-canvas").style.width, "200px");
    assert.equal(e.element("canvas-zoom-label").textContent, "play · 200%");
    e.run("reviewSuggestion(0,true); fitEditorImage();");
    assert.equal(e.run("currentAnnotations[0].xyxy.join(',')"), "45,75,81,120");
    assert.equal(e.element("annotation-canvas").style.width, "");
  });
  await check("empty automatic recovery leaves an approved capture clean", async () => {
    const e = editor();
    e.context.fetch = async () => ({ok:true,json:async()=>({proposals:[],warnings:[]})});
    e.run(`allRecords[0].curation_status="approved"; allRecords[0].has_accessibility_snapshot=true; openEditor("sample");`);
    await new Promise(setImmediate);
    assert.equal(e.run("editorDirty"), false);
    assert.equal(e.element("curation-status-select").value, "approved");
  });
  await check("a response cannot resurrect a native candidate accepted during automatic detection", async () => {
    const e = editor();
    let respond;
    e.context.fetch = () => new Promise(resolve => {respond=resolve;});
    e.run(`allRecords[0].display_annotations[0].review_required=true; allRecords[0].has_accessibility_snapshot=true; openEditor("sample"); reviewSuggestion(0,true);`);
    respond({ok:true,json:async()=>({proposals:[{class:"save",xyxy:[10,20,40,60],proposal_id:"same-save"}],warnings:[]})});
    await new Promise(setImmediate);
    assert.equal(e.run("currentProposals.length"), 0);
    assert.equal(e.run("currentAnnotations.length"), 1);
    assert.equal(e.run("proposalDecisions[0].decision"), "accepted");
  });
  await check("automatic results cannot attach to an editor closed while detection ran", async () => {
    const e = editor();
    let respond;
    e.context.fetch = () => new Promise(resolve => {respond=resolve;});
    e.run(`allRecords[0].has_accessibility_snapshot=true; openEditor("sample"); closeEditor(true);`);
    respond({ok:true,json:async()=>({proposals:[{class:"playhead",xyxy:[60,70,90,100]}],warnings:[]})});
    await new Promise(setImmediate);
    assert.equal(e.run("activeRecord"), null);
    assert.equal(e.run("currentProposals.length"), 0);
  });
  await check("refining preserves native candidates when no replacement is found", async () => {
    const e = editor();
    e.run(`allRecords[0].display_annotations[0].review_required=true; openEditor("sample");`);
    let input;
    e.context.fetch = async (url, options) => {
      input = JSON.parse(options.body);
      return {ok: true, json: async () => ({proposals: [], warnings: []})};
    };
    await e.run("refineIconBoxes()");
    assert.equal(input.annotations.length, 1);
    assert.equal(e.run("currentProposals.length"), 1);
    assert.equal(e.run("proposalDecisions[0].decision"), "pending");
  });
  await check("refining a pending native box accepts its replacement exactly once", async () => {
    const e = editor();
    e.run(`allRecords[0].display_annotations[0].review_required=true; openEditor("sample");`);
    e.context.fetch = async () => ({ok: true, json: async () => ({proposals: [{class:"save",xyxy:[15,25,35,50],action:"replace_geometry",replaces:{class:"save",xyxy:[10,20,40,60]},proposal_id:"native-refine"}], warnings: []})});
    await e.run("refineIconBoxes()");
    e.run("reviewSuggestion(1,true)");
    assert.equal(e.run("currentAnnotations.length"), 1);
    assert.equal(e.run("currentAnnotations[0].xyxy.join(',')"), "15,25,35,50");
    assert.equal(e.run("currentProposals.length"), 0);
    assert.equal(e.run("proposalDecisions.map(p=>p.decision).join(',')"), "superseded,accepted");
  });
  await check("auto capture stops on failure instead of repeatedly taking screenshots", async () => {
    const e = editor();
    e.context.fetch = async (url) => ({ok: true, json: async () => url === "/api/records" ? [] : ({success:false,message:"No new capture was published",log:"run=1 fail=0"})});
    e.run("isAutoCapturing=true");
    await e.run("runAutoCaptureIteration()");
    assert.equal(e.run("isAutoCapturing"), false);
    assert.match(e.element("capture-status-title").textContent, /No new capture/);
    assert.equal(e.element("capture-log").textContent, "run=1 fail=0");
  });
  await check("icon refinement replaces the original tap box and keeps its provenance", () => {
    const e = editor();
    e.run(`currentProposals=[{class:"save",xyxy:[15,25,35,50],visual_xyxy:[15,25,35,50],interaction_xyxy:[10,20,40,60],action:"replace_geometry",replaces:{class:"save",xyxy:[10,20,40,60]},proposal_id:"refine",geometry_policy:"icon_only",review_required:true}]; proposalDecisions=currentProposals.map(p=>({...p,decision:"pending"})); reviewSuggestion(0,true);`);
    assert.equal(e.run("currentAnnotations.length"), 1);
    assert.equal(e.run("currentAnnotations[0].xyxy.join(',')"), "15,25,35,50");
    assert.equal(e.run("currentAnnotations[0].interaction_xyxy.join(',')"), "10,20,40,60");
    assert.equal(e.run("currentAnnotations[0].geometry_reviewed"), true);
  });
  await check("a stale refinement cannot replace a box changed while the request ran", () => {
    const e = editor();
    e.run(`currentAnnotations[0].xyxy=[11,20,40,60]; currentProposals=[{class:"save",xyxy:[15,25,35,50],action:"replace_geometry",replaces:{class:"save",xyxy:[10,20,40,60]},proposal_id:"refine"}]; reviewSuggestion(0,true);`);
    assert.match(e.alerts[0], /original box changed/);
    assert.equal(e.run("currentAnnotations[0].xyxy.join(',')"), "11,20,40,60");
    assert.equal(e.run("currentProposals.length"), 1);
  });
  await check("manual edits preserve tap regions and the Subscribe / Skip Ad policies", () => {
    const e = editor();
    e.run("editAnnotation(0)");
    e.element("draw-class-select").value = "skip_ad";
    e.run("applyBoxChanges()");
    assert.equal(e.run("currentAnnotations[0].geometry_policy"), "text_and_icon");
    assert.equal(e.run("currentAnnotations[0].interaction_xyxy.join(',')"), "10,20,40,60");
    e.element("draw-class-select").value = "subscribe";
    e.run("applyBoxChanges()");
    assert.equal(e.run("currentAnnotations[0].geometry_policy"), "visible_button");
  });
  await check("approval is blocked while suggestions remain pending", async () => {
    const e = editor();
    e.run(`currentProposals=[{class:"like",xyxy:[60,70,90,100]}];`);
    e.element("curation-status-select").value = "approved";
    await e.run("saveCuration()");
    assert.match(e.alerts[0], /remaining suggestions/);
    assert.equal(e.run("activeRecord.record_id"), "sample");
  });
  await check("regenerating suggestions supersedes old pending decisions", async () => {
    const e = editor();
    e.run(`proposalDecisions=[{proposal_id:"old",decision:"pending"}];`);
    e.context.fetch = async () => ({ok: true, json: async () => ({proposals: [{class: "like", xyxy: [60,70,90,100], proposal_id: "new"}], warnings: []})});
    await e.run("suggestAnnotations()");
    assert.equal(e.run("proposalDecisions[0].decision"), "superseded");
    assert.equal(e.run("proposalDecisions[1].decision"), "pending");
  });
  await check("late responses cannot attach suggestions to a closed editor", async () => {
    const e = editor();
    let respond;
    e.context.fetch = () => new Promise((resolve) => {respond = resolve;});
    const pending = e.run("suggestAnnotations()");
    e.run("closeEditor(true)");
    respond({ok: true, json: async () => ({proposals: [{class:"like",xyxy:[0,0,20,20]}], warnings: []})});
    await pending;
    assert.equal(e.run("activeRecord"), null);
    assert.equal(e.run("currentProposals.length"), 0);
  });
  await check("saving preserves review decisions and sends original pixel coordinates", async () => {
    const e = editor();
    const posts = [];
    e.context.fetch = async (url, options) => {
      if (options?.method === "POST") posts.push(JSON.parse(options.body));
      return {ok: true, json: async () => []};
    };
    e.run(`proposalDecisions=[{proposal_id:"checked",decision:"rejected"}];`);
    await e.run("saveCuration()");
    assert.deepEqual(posts[0].annotations[0].xyxy, [10,20,40,60]);
    assert.equal(posts[0].reviewed_proposals[0].decision, "rejected");
    assert.equal(posts[0].status, "needs_review");
    assert.equal(e.run("activeRecord"), null);
  });
  console.log(`${passed} editor interaction checks passed.`);
}

module.exports = {editor};
if (require.main === module) main().catch((error) => {console.error(error); process.exitCode = 1;});
