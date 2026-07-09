(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var danger = style.getPropertyValue('--danger').trim();
  var success = style.getPropertyValue('--success').trim();

  // === Chart 1: Closure Validation — Before/After ===
  var chart1 = echarts.init(document.getElementById('chart-closure'), null, { renderer: 'svg' });
  chart1.setOption({
    animation: false,
    tooltip: { trigger: 'axis', appendToBody: true },
    legend: { data: ['校正前 a/b', '校正后 a/b', '理想值 (1.0)'], textStyle: { color: muted }, top: 0 },
    grid: { left: '8%', right: '5%', bottom: '10%', top: '15%' },
    xAxis: {
      type: 'category',
      data: ['0°', '3°', '5°', '8°', '10°', '12°', '15°', '20°'],
      name: '倾角 θ',
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted }
    },
    yAxis: {
      type: 'value',
      name: 'a/b 比值',
      min: 0.99,
      max: 1.08,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: '校正前 a/b',
        type: 'bar',
        data: [1.0006, 1.0014, 1.0039, 1.0055, 1.0051, 1.0123, 1.0213, 1.0592],
        itemStyle: { color: accent2 },
        barWidth: '30%'
      },
      {
        name: '校正后 a/b',
        type: 'bar',
        data: [1.0004, 1.0004, 1.0004, 1.0008, 1.0007, 1.0008, 1.0009, 1.0017],
        itemStyle: { color: success },
        barWidth: '30%'
      },
      {
        name: '理想值 (1.0)',
        type: 'line',
        data: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        itemStyle: { color: muted },
        lineStyle: { type: 'dashed', width: 2 },
        symbol: 'none'
      }
    ]
  });
  window.addEventListener('resize', function() { chart1.resize(); });

  // === Chart 2: Scale vs Center — 6 schemes ===
  var chart2 = echarts.init(document.getElementById('chart-scale-center'), null, { renderer: 'svg' });
  chart2.setOption({
    animation: false,
    tooltip: { trigger: 'axis', appendToBody: true },
    legend: {
      data: ['GT中心+GTscale 各向异性', 'GT中心+GTscale naive', '检测中心+检测scale 各向异性', '检测中心+检测scale naive', '理论naive误差'],
      textStyle: { color: muted, fontSize: 11 },
      top: 0,
      type: 'scroll'
    },
    grid: { left: '8%', right: '5%', bottom: '10%', top: '20%' },
    xAxis: {
      type: 'category',
      data: ['0°', '3°', '5°', '8°', '10°', '12°', '15°', '20°'],
      name: '倾角 θ',
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted }
    },
    yAxis: {
      type: 'value',
      name: '振幅误差 (%)',
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: 'GT中心+GTscale 各向异性',
        type: 'line',
        data: [0.03, 0.13, 0.38, 1.03, 1.62, 2.33, 3.66, 6.16],
        itemStyle: { color: accent },
        lineStyle: { width: 2 }
      },
      {
        name: 'GT中心+GTscale naive',
        type: 'line',
        data: [0.03, 0.11, 0.32, 0.85, 1.33, 1.91, 2.97, 5.49],
        itemStyle: { color: accent2 },
        lineStyle: { width: 2 }
      },
      {
        name: '检测中心+检测scale 各向异性',
        type: 'line',
        data: [7.42, 7.66, 8.14, 8.55, 9.32, 9.53, 10.03, 10.40],
        itemStyle: { color: danger },
        lineStyle: { width: 1.5, type: 'dashed' }
      },
      {
        name: '检测中心+检测scale naive',
        type: 'line',
        data: [7.41, 7.62, 8.08, 8.47, 9.45, 9.60, 10.27, 10.82],
        itemStyle: { color: muted },
        lineStyle: { width: 1.5, type: 'dashed' }
      },
      {
        name: '理论naive误差',
        type: 'line',
        data: [0, 0.14, 0.38, 0.98, 1.54, 2.22, 3.48, 6.16],
        itemStyle: { color: success },
        lineStyle: { width: 1, type: 'dotted' },
        symbol: 'none'
      }
    ]
  });
  window.addEventListener('resize', function() { chart2.resize(); });

  // === Chart 3: Root Cause — Scale Comparison ===
  var chart3 = echarts.init(document.getElementById('chart-root-cause'), null, { renderer: 'svg' });
  chart3.setOption({
    animation: false,
    tooltip: { trigger: 'axis', appendToBody: true },
    legend: { data: ['D/(2a) (当前)', 'Z₀/F (正确V方向)', 'D/(2b) (U方向,正确)'], textStyle: { color: muted }, top: 0 },
    grid: { left: '10%', right: '5%', bottom: '10%', top: '15%' },
    xAxis: {
      type: 'category',
      data: ['0°', '5°', '10°', '15°', '20°'],
      name: '倾角 θ',
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted }
    },
    yAxis: {
      type: 'value',
      name: 'Scale (μm/px)',
      min: 70,
      max: 90,
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: 'D/(2a) (当前)',
        type: 'bar',
        data: [85.4, 84.7, 83.0, 79.9, 75.4],
        itemStyle: { color: danger },
        barWidth: '25%'
      },
      {
        name: 'Z₀/F (正确V方向)',
        type: 'line',
        data: [85.4, 85.4, 85.4, 85.4, 85.4],
        itemStyle: { color: accent2 },
        lineStyle: { width: 2, type: 'dashed' },
        symbolSize: 8
      },
      {
        name: 'D/(2b) (U方向,正确)',
        type: 'line',
        data: [85.4, 85.1, 84.2, 82.0, 80.2],
        itemStyle: { color: success },
        lineStyle: { width: 2 },
        symbolSize: 8
      }
    ]
  });
  window.addEventListener('resize', function() { chart3.resize(); });

  // === Chart 4: Perspective Corrected Scale Verification ===
  var chart4 = echarts.init(document.getElementById('chart-persp-verify'), null, { renderer: 'svg' });
  chart4.setOption({
    animation: false,
    tooltip: { trigger: 'axis', appendToBody: true },
    legend: { data: ['透视修正 D·a/(2b²)', 'naive D/(2√ab)', '各向异性 D/(2a)', '理论 naive 误差'], textStyle: { color: muted }, top: 0 },
    grid: { left: '8%', right: '5%', bottom: '10%', top: '15%' },
    xAxis: {
      type: 'category',
      data: ['0°', '3°', '5°', '8°', '10°', '12°', '15°', '20°'],
      name: '倾角 θ',
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted }
    },
    yAxis: {
      type: 'value',
      name: '振幅误差 (%)',
      axisLine: { lineStyle: { color: rule } },
      axisLabel: { color: muted },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: '透视修正 D·a/(2b²)',
        type: 'line',
        data: [0.000, 0.656, 1.012, 1.434, 1.646, 1.805, 1.950, 3.707],
        itemStyle: { color: success },
        lineStyle: { width: 3 },
        symbolSize: 8
      },
      {
        name: 'naive D/(2√ab)',
        type: 'line',
        data: [0.729, 1.362, 1.684, 2.099, 3.009, 2.879, 3.447, 5.608],
        itemStyle: { color: accent2 },
        lineStyle: { width: 2 }
      },
      {
        name: '各向异性 D/(2a)',
        type: 'line',
        data: [0.722, 1.354, 1.662, 2.074, 2.960, 2.803, 3.368, 5.252],
        itemStyle: { color: accent },
        lineStyle: { width: 2 }
      },
      {
        name: '理论 naive 误差',
        type: 'line',
        data: [0, 0.069, 0.191, 0.490, 0.768, 1.111, 1.749, 3.159],
        itemStyle: { color: muted },
        lineStyle: { width: 1, type: 'dotted' },
        symbol: 'none'
      }
    ]
  });
  window.addEventListener('resize', function() { chart4.resize(); });

})();
