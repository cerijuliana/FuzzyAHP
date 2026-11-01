document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.toggle-password').forEach(function (button) {
    button.addEventListener('click', function () {
      const wrapper = button.closest('.password-wrap');
      const input = wrapper ? wrapper.querySelector('input') : null;
      if (!input) return;

      const isHidden = input.type === 'password';
      input.type = isHidden ? 'text' : 'password';
      button.textContent = isHidden ? 'Sembunyi' : 'Lihat';
      button.setAttribute('aria-label', isHidden ? 'Sembunyikan password' : 'Tampilkan password');
    });
  });

  const RI_TABLE = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59
  };

  function parseItems(form) {
    try {
      return JSON.parse(form.dataset.items || '[]');
    } catch (e) {
      return [];
    }
  }

  function buildMatrix(items, form) {

    const n = items.length;
    const index = Object.fromEntries(items.map((item, i) => [item, i]));
    const matrix = Array.from(
        { length: n },
        () => Array.from({ length: n }, () => 1)
    );

    const fd = new FormData(form);

    // Kelompokkan semua expert berdasarkan pair
    const pairMap = {};

    for (const [name, value] of fd.entries()) {

        if (!name.startsWith("winner_")) continue;

        // contoh:
        // winner_K1|K2
        // winner_K1|K2_0
        // winner_K1|K2_1

        let key = name.substring(7);

        let expertsIndex = null;

        const m = key.match(/(.+)_(\d+)$/);

        if (m) {
            key = m[1];
            expertsIndex = m[2];
        }

        if (!(key in pairMap))
            pairMap[key] = [];

        let scaleName = "scale_" + key;

        if (expertsIndex !== null)
            scaleName += "_" + expertsIndex;

        const scale = Number(fd.get(scaleName) || 1);

        pairMap[key].push({
            winner: value,
            scale: Math.max(1, Math.min(9, scale))
        });
    }

    // Bangun matriks
    for (const key in pairMap) {

        const [a, b] = key.split("|");

        if (!(a in index) || !(b in index))
            continue;

        const values = [];

        pairMap[key].forEach(v => {

            if (v.winner === a)
                values.push(v.scale);
            else
                values.push(1 / v.scale);

        });

        // geometric mean
        let gm = 1;

        values.forEach(v => gm *= v);

        gm = Math.pow(gm, 1 / values.length);

        const i = index[a];
        const j = index[b];

        matrix[i][j] = gm;
        matrix[j][i] = 1 / gm;
    }

    return matrix;
}

  function principalEigenApprox(matrix) {
    const n = matrix.length;
    if (n === 0) return { lambdaMax: 0, weights: [] };
    let w = Array.from({ length: n }, () => 1 / n);

    for (let iter = 0; iter < 100; iter++) {
      const next = Array.from({ length: n }, (_, i) => {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += matrix[i][j] * w[j];
        return Math.abs(sum);
      });
      const total = next.reduce((a, b) => a + b, 0) || 1;
      w = next.map(x => x / total);
    }

    const Aw = Array.from({ length: n }, (_, i) => {
      let sum = 0;
      for (let j = 0; j < n; j++) sum += matrix[i][j] * w[j];
      return sum;
    });
    const ratios = Aw.map((x, i) => w[i] === 0 ? 0 : x / w[i]);
    const lambdaMax = ratios.reduce((a, b) => a + b, 0) / n;
    return { lambdaMax, weights: w };
  }

  function calculateConsistency(matrix) {
    const n = matrix.length;
    if (n <= 2) return { lambdaMax: n, ci: 0, cr: 0 };
    const { lambdaMax } = principalEigenApprox(matrix);
    const ci = (lambdaMax - n) / (n - 1);
    const ri = RI_TABLE[n] || 1.59;
    const cr = ri > 0 ? ci / ri : 0;
    return { lambdaMax, ci, cr };
  }

  function renderMatrix(items, matrix, tbody) {
    if (!tbody) return;
    tbody.innerHTML = '';
    matrix.forEach((row, i) => {
      const tr = document.createElement('tr');
      const label = document.createElement('td');
      label.textContent = items[i];
      tr.appendChild(label);
      row.forEach(value => {
        const td = document.createElement('td');
        td.textContent = Number(value).toFixed(4);
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
  }

  function renderConsistency(result, target) {
    if (!target) return;
    const okClass = result.cr <= 0.10 ? 'ok' : 'bad';
    target.innerHTML = '<b>Uji Konsistensi:</b> '
      + 'λmax=' + result.lambdaMax.toFixed(4)
      + ' | CI=' + result.ci.toFixed(4)
      + ' | CR=<span class="' + okClass + '">' + result.cr.toFixed(4) + '</span> '
      + '<span class="muted">(Valid jika ≤ 0,10)</span>';
  }

  function updateLiveMatrix(form) {
    const items = parseItems(form);
    if (items.length < 2) return;
    const matrix = buildMatrix(items, form);
    const tbody = document.getElementById(form.dataset.matrixTarget || '');
    const target = document.getElementById(form.dataset.consistencyTarget || '');
    renderMatrix(items, matrix, tbody);
    renderConsistency(calculateConsistency(matrix), target);
  }

  document.querySelectorAll('.js-live-matrix-form').forEach(function (form) {
    updateLiveMatrix(form);
    form.querySelectorAll('select[name^="winner_"], select[name^="scale_"]').forEach(function (select) {
      select.addEventListener('change', function () {
        updateLiveMatrix(form);
      });
    });
  });
});
