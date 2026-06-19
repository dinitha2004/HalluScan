import React from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

// Plots fused hallucination RISK per turn (higher = worse -> red at the top).
const RiskChart = ({ data }) => {
    const pts = (data || []).filter(d => d.risk != null);
    if (pts.length === 0) return <div className="flex h-full items-center justify-center text-gray-400">No data available</div>;

    const chartData = {
        labels: pts.map((_, i) => `Q${i + 1}`),
        datasets: [{
            label: 'Hallucination risk',
            data: pts.map(d => d.risk ?? 0),
            borderColor: '#ff3b30',
            backgroundColor: (ctx) => {
                const { ctx: c, chartArea } = ctx.chart;
                if (!chartArea) return 'rgba(255,59,48,0.1)';
                const g = c.createLinearGradient(0, chartArea.bottom, 0, chartArea.top);
                g.addColorStop(0, 'rgba(52,199,89,0.15)');    // green at bottom (low risk)
                g.addColorStop(0.5, 'rgba(255,179,0,0.12)');  // amber middle
                g.addColorStop(1, 'rgba(255,59,48,0.18)');    // red at top (high risk)
                return g;
            },
            tension: 0.4,
            fill: true,
            pointBackgroundColor: (ctx) => {
                const v = ctx.raw;
                if (v >= 0.5) return '#ff3b30';
                if (v >= 0.33) return '#ffb300';
                return '#34c759';
            },
            pointBorderColor: '#fff', pointBorderWidth: 2, pointRadius: 5, pointHoverRadius: 7,
        }],
    };

    const options = {
        responsive: true, maintainAspectRatio: false,
        plugins: {
            legend: { position: 'top', align: 'end', labels: { boxWidth: 10, usePointStyle: true, font: { family: 'Inter', size: 11 } } },
            tooltip: {
                mode: 'index', intersect: false, backgroundColor: 'rgba(255,255,255,0.95)',
                titleColor: '#000', bodyColor: '#666', borderColor: '#ddd', borderWidth: 1, padding: 12, displayColors: false,
                callbacks: {
                    title: (items) => pts[items[0].dataIndex].question,
                    label: (item) => {
                        const v = item.raw, pct = (v * 100).toFixed(1);
                        const label = v >= 0.5 ? 'Likely Hallucinated' : v >= 0.33 ? 'Uncertain' : 'Reliable';
                        return `Risk: ${pct}% (${label})`;
                    },
                },
            },
        },
        scales: {
            y: { beginAtZero: true, max: 1.0, grid: { color: '#f0f0f0' }, ticks: { font: { family: 'Inter', size: 10 }, callback: (v) => `${(v * 100).toFixed(0)}%` } },
            x: { grid: { display: false }, ticks: { font: { family: 'Inter', size: 10 } } },
        },
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
    };

    return <Line data={chartData} options={options} />;
};

export default RiskChart;
