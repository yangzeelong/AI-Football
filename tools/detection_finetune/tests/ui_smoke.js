// 复核界面前端冒烟测试：抓取服务端渲染后的页面，用最小 DOM 桩把内嵌 JS 跑一遍。
// 能抓到的典型问题：函数用了没定义（如 boxesOf）、初始化抛异常、绑定到不存在的元素。
const URL_ = process.argv[2] || 'http://127.0.0.1:8090/';

const html = await (await fetch(URL_)).text();
const js = html.split('<script>')[1].split('</script>')[0];

const ctxStub = new Proxy({}, { get: (_, k) => (k === 'canvas' ? {} : () => {}) });
const made = {};
function el(id) {
  if (!made[id]) {
    made[id] = {
      id, textContent: '', innerHTML: '', value: '', checked: false, style: {}, children: [],
      classList: { add() {}, remove() {}, contains: () => false },
      appendChild(c) { this.children.push(c); }, removeChild() {},
      getContext: () => ctxStub, getBoundingClientRect: () => ({ left: 0, top: 0, width: 960, height: 540 }),
      clientWidth: 1000, clientHeight: 600, naturalWidth: 1280, naturalHeight: 720,
      addEventListener() {}, remove() {}, onclick: null, onchange: null, src: '',
    };
  }
  return made[id];
}
const store = {};
const documentStub = {
  getElementById: el,
  createElement: (tag) => el('new_' + tag + '_' + Math.random().toString(36).slice(2, 6)),
  querySelector: () => null,
  onkeydown: null,
  execCommand() {},
};
const windowStub = {};
const localStorageStub = {
  getItem: (k) => (k in store ? store[k] : null),
  setItem: (k, v) => { store[k] = String(v); },
  removeItem: (k) => { delete store[k]; },
};
class ImageStub { set src(v) { if (this.onload) this.onload(); } }

const errors = [];
try {
  // eslint-disable-next-line no-new-func
  new Function('document', 'window', 'localStorage', 'Image', 'fetch', 'console', js)(
    documentStub, windowStub, localStorageStub, ImageStub, async () => ({}), console);
} catch (err) {
  errors.push('执行页面 JS 抛异常：' + err.message);
}

// 断言：初始化后关键元素被填过
const checks = [
  ['位置指示被写入', made['pos'] && made['pos'].textContent.includes('/')],
  ['批次下拉有内容', made['bsel'] && made['bsel'].children.length > 0],
  // build 号只要是"非空且不是占位符 ?"即可（别把版本名写死，否则改版本号就会误报）
  ['build 号已注入', !!made['build'] && made['build'].textContent.length > 0
                     && made['build'].textContent !== '?'],
  ['保存按钮已绑定', made['save'] && typeof made['save'].onclick === 'function'],
  ['复用按钮已绑定', made['reuse'] && typeof made['reuse'].onclick === 'function'],
  ['标记按钮已绑定', made['mark'] && typeof made['mark'].onclick === 'function'],
];
for (const [name, ok] of checks) if (!ok) errors.push('未通过：' + name);

if (errors.length) {
  console.log('前端冒烟：失败');
  for (const e of errors) console.log('  ✗ ' + e);
  process.exit(1);
}
console.log('前端冒烟：通过（%d 项断言；批次 %d 个）', checks.length, made['bsel'].children.length);
