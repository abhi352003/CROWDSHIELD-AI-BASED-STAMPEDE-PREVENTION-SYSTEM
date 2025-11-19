// Live data updates via Server-Sent Events
const evtSource = new EventSource("/data_feed");

// Initialize Chart.js for live crowd trend
const ctx = document.getElementById("crowdChart").getContext("2d");
const crowdChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Crowd Count",
        borderColor: "#00ffcc",
        data: [],
        fill: false,
        tension: 0.2,
      },
      {
        label: "Violations",
        borderColor: "#ffcc00",
        data: [],
        fill: false,
        tension: 0.2,
      },
    ],
  },
  options: {
    responsive: true,
    scales: {
      x: { ticks: { color: "#fff" } },
      y: { ticks: { color: "#fff" } },
    },
    plugins: {
      legend: { labels: { color: "#fff" } },
    },
  },
});

// Handle incoming real-time data
evtSource.onmessage = function (event) {
  const data = JSON.parse(event.data);
  const crowd = data.crowd || {};

  // Update text stats
  document.getElementById("time").innerText = crowd.time || "--";
  document.getElementById("crowd").innerText = crowd.human_count || 0;
  document.getElementById("violations").innerText = crowd.violation_count || 0;
  document.getElementById("restricted").innerText = crowd.restricted_entry ? "🚫 Yes" : "✅ No";
  document.getElementById("abnormal").innerText = crowd.abnormal_activity ? "⚠️ Detected" : "Normal";

  const abnormalBox = document.getElementById("abnormal");
  abnormalBox.style.color = crowd.abnormal_activity ? "red" : "white";

  // Update chart data
  const timeLabel = new Date().toLocaleTimeString();
  crowdChart.data.labels.push(timeLabel);
  crowdChart.data.datasets[0].data.push(crowd.human_count);
  crowdChart.data.datasets[1].data.push(crowd.violation_count);

  if (crowdChart.data.labels.length > 15) {
    crowdChart.data.labels.shift();
    crowdChart.data.datasets.forEach((ds) => ds.data.shift());
  }

  crowdChart.update();

  // Refresh analysis images
 // 🔁 Refresh analysis images every 8 seconds (instead of every second)
setInterval(() => {
  document.getElementById("heatmapImg").src = `/heatmap?rand=${Math.random()}`;
  document.getElementById("crowdPlotImg").src = `/crowd_plot?rand=${Math.random()}`;
  document.getElementById("energyImg").src = `/energy_plot?rand=${Math.random()}`;
}, 8000); // 8 seconds refresh

};
