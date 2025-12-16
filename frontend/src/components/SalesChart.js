import React, { useState } from "react";

const WIDTH = 900;
const HEIGHT = 400;
const PADDING = 70;

const SalesChart = ({ data, title }) => {
  const [chartType, setChartType] = useState("bar");

  if (!data || data.length === 0) return null;

  const maxValue = Math.max(...data.map(d => d.amount));

  const months = data.map(d => d.month);
  const current = data.filter(d => d.type === "Actual");
  const predicted = data.filter(d => d.type === "Predicted");

  /* ---------------- PIE CHART ---------------- */
  const total = data.reduce((sum, d) => sum + d.amount, 0);
  let startAngle = 0;

  const pieSlices = data.map((d, i) => {
    const percentage = ((d.amount / total) * 100).toFixed(1);
    const angle = (d.amount / total) * 360;
    const endAngle = startAngle + angle;
    const largeArc = angle > 180 ? 1 : 0;

    const x1 = 100 + 80 * Math.cos((Math.PI * startAngle) / 180);
    const y1 = 100 + 80 * Math.sin((Math.PI * startAngle) / 180);
    const x2 = 100 + 80 * Math.cos((Math.PI * endAngle) / 180);
    const y2 = 100 + 80 * Math.sin((Math.PI * endAngle) / 180);

    const path = `
      M100,100
      L${x1},${y1}
      A80,80 0 ${largeArc} 1 ${x2},${y2}
      Z
    `;

    startAngle += angle;

    return {
      path,
      color: `hsl(${i * 40}, 70%, 60%)`,
      label: `${d.month} ${d.year}`,
      value: d.amount,
      percentage: percentage
    };
  });

  /* ---------------- LINE GRAPH ---------------- */
  // Sort data chronologically for proper line plotting
  const monthOrder = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const sortedData = [...data].sort((a, b) => {
    if (a.year !== b.year) return a.year - b.year;
    return monthOrder.indexOf(a.month) - monthOrder.indexOf(b.month);
  });

  const scaleX = (WIDTH - PADDING * 2) / (sortedData.length - 1 || 1);
  const scaleY = (HEIGHT - PADDING * 2) / maxValue;

  const drawLine = (dataset, color) => {
    // Sort dataset chronologically
    const sortedDataset = [...dataset].sort((a, b) => {
      if (a.year !== b.year) return a.year - b.year;
      return monthOrder.indexOf(a.month) - monthOrder.indexOf(b.month);
    });

    return (
    <>
      <polyline
        fill="none"
        stroke={color}
          strokeWidth="2.5"
          strokeOpacity="0.9"
          points={sortedDataset
            .map((d) => {
              const index = sortedData.findIndex(item =>
                item.month === d.month &&
                item.year === d.year &&
                item.type === d.type
              );
              // If exact match not found, find by month and year only
              const fallbackIndex = index >= 0 ? index : sortedData.findIndex(item =>
                item.month === d.month && item.year === d.year
              );
              const finalIndex = fallbackIndex >= 0 ? fallbackIndex : 0;
              const x = PADDING + finalIndex * scaleX;
            const y = HEIGHT - PADDING - d.amount * scaleY;
            return `${x},${y}`;
          })
          .join(" ")}
      />
        {sortedDataset.map((d, i) => {
          const index = sortedData.findIndex(item =>
            item.month === d.month &&
            item.year === d.year &&
            item.type === d.type
          );
          const fallbackIndex = index >= 0 ? index : sortedData.findIndex(item =>
            item.month === d.month && item.year === d.year
          );
          const finalIndex = fallbackIndex >= 0 ? fallbackIndex : 0;
          const x = PADDING + finalIndex * scaleX;
        const y = HEIGHT - PADDING - d.amount * scaleY;
        return <circle key={i} cx={x} cy={y} r="4" fill={color} />;
      })}
    </>
  );
  };

  // Prepare grouped periods (one group per month+year) for bar chart
  const periodMap = {};
  sortedData.forEach((d) => {
    const key = `${d.month}-${d.year}`;
    if (!periodMap[key]) {
      periodMap[key] = { month: d.month, year: d.year, actual: null, predicted: null };
    }
    if (d.type === "Actual") periodMap[key].actual = d.amount;
    if (d.type === "Predicted") periodMap[key].predicted = d.amount;
  });

  const periods = Object.values(periodMap);
  const groupCount = periods.length || 1;
  const availableWidth = WIDTH - PADDING * 2;
  const groupSpacing = availableWidth / groupCount;
  const groupWidth = Math.max(12, Math.min(60, groupSpacing * 0.6));

  // Compute simple linear regression (least squares) for Actual values across period indices
  const actualPoints = periods
    .map((p, i) => ({ x: i, y: p.actual }))
    .filter(pt => pt.y != null);

  let trendLine = null;
  if (actualPoints.length >= 2) {
    const n = actualPoints.length;
    const sumX = actualPoints.reduce((s, p) => s + p.x, 0);
    const sumY = actualPoints.reduce((s, p) => s + p.y, 0);
    const sumXY = actualPoints.reduce((s, p) => s + p.x * p.y, 0);
    const sumX2 = actualPoints.reduce((s, p) => s + p.x * p.x, 0);
    const denom = n * sumX2 - sumX * sumX;
    const m = denom === 0 ? 0 : (n * sumXY - sumX * sumY) / denom;
    const b = (sumY - m * sumX) / n;

    const xA = 0;
    const xB = groupCount - 1;
    const x1 = PADDING + (xA + 0.5) * groupSpacing;
    const y1 = HEIGHT - PADDING - (m * xA + b) * scaleY;
    const x2 = PADDING + (xB + 0.5) * groupSpacing;
    const y2 = HEIGHT - PADDING - (m * xB + b) * scaleY;
    trendLine = { x1, y1, x2, y2 };
  }

  // Compute regression for Predicted values as well
  const predictedPointsForTrend = periods
    .map((p, i) => ({ x: i, y: p.predicted }))
    .filter(pt => pt.y != null);

  let trendLinePred = null;
  if (predictedPointsForTrend.length >= 2) {
    const n2 = predictedPointsForTrend.length;
    const sumX2_ = predictedPointsForTrend.reduce((s, p) => s + p.x, 0);
    const sumY2 = predictedPointsForTrend.reduce((s, p) => s + p.y, 0);
    const sumXY2 = predictedPointsForTrend.reduce((s, p) => s + p.x * p.y, 0);
    const sumX22 = predictedPointsForTrend.reduce((s, p) => s + p.x * p.x, 0);
    const denom2 = n2 * sumX22 - sumX2_ * sumX2_;
    const m2 = denom2 === 0 ? 0 : (n2 * sumXY2 - sumX2_ * sumY2) / denom2;
    const b2 = (sumY2 - m2 * sumX2_) / n2;

    const xa = 0;
    const xb = groupCount - 1;
    const xx1 = PADDING + (xa + 0.5) * groupSpacing;
    const yy1 = HEIGHT - PADDING - (m2 * xa + b2) * scaleY;
    const xx2 = PADDING + (xb + 0.5) * groupSpacing;
    const yy2 = HEIGHT - PADDING - (m2 * xb + b2) * scaleY;
    trendLinePred = { x1: xx1, y1: yy1, x2: xx2, y2: yy2 };
  }

  return (
    <div className="card p-3">
      {/* Header */}
      <div className="d-flex justify-content-between mb-3">
        <h5>{title}</h5>
        <select
          className="form-select w-auto"
          value={chartType}
          onChange={(e) => setChartType(e.target.value)}
        >
          <option value="bar"> Bar Graph</option>
          <option value="pie"> Pie Chart</option>
          <option value="line"> Line Graph</option>
        </select>
      </div>

      {/* BAR GRAPH */}
      {chartType === "bar" && (
        <>
          <div style={{ overflowX: "auto", overflowY: "visible", width: "100%", paddingBottom: "20px" }}>
            <svg width={WIDTH} height={HEIGHT + 80} style={{ display: "block", minWidth: "100%" }}>
              {/* Axes */}
              <line x1={PADDING} y1={HEIGHT - PADDING} x2={WIDTH - PADDING} y2={HEIGHT - PADDING} stroke="black" />
              <line x1={PADDING} y1={PADDING} x2={PADDING} y2={HEIGHT - PADDING} stroke="black" />

              {/* Y-Axis ticks */}
              {[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((ratio, i) => {
                const value = Math.round(maxValue * ratio);
                const y = HEIGHT - PADDING - value * scaleY;
                return (
                  <g key={i}>
                    <line x1={PADDING - 5} y1={y} x2={PADDING} y2={y} stroke="black" strokeWidth={ratio === 0 || ratio === 0.5 || ratio === 1.0 ? "1.5" : "0.5"} />
                    <text x={PADDING - 15} y={y + 4} fontSize="9" textAnchor="end">{value}</text>
                  </g>
                );
              })}

              {/* Bars per period (grouped by month-year) */}
              {periods.map((p, i) => {
                const centerX = PADDING + (i + 0.5) * groupSpacing;
                const hasBoth = p.actual != null && p.predicted != null;
                const barW = hasBoth ? groupWidth / 2 : groupWidth;
                const leftStart = centerX - (hasBoth ? groupWidth / 2 : barW / 2);
                const actualH = p.actual != null ? p.actual * scaleY : 0;
                const predictedH = p.predicted != null ? p.predicted * scaleY : 0;
                return (
                  <g key={i}>
                    {p.actual != null && (
                      <rect x={leftStart} y={HEIGHT - PADDING - actualH} width={barW} height={actualH} fill="blue" opacity="0.9" />
                    )}
                    {p.predicted != null && (
                      <rect x={leftStart + (hasBoth ? barW : 0)} y={HEIGHT - PADDING - predictedH} width={barW} height={predictedH} fill="green" opacity="0.8" />
                    )}
                  </g>
                );
              })}

              {/* Trend lines */}
              {trendLine && (
                <line x1={trendLine.x1} y1={trendLine.y1} x2={trendLine.x2} y2={trendLine.y2} stroke="red" strokeWidth="2.5" strokeDasharray="6 4" />
              )}
              {trendLinePred && (
                <line x1={trendLinePred.x1} y1={trendLinePred.y1} x2={trendLinePred.x2} y2={trendLinePred.y2} stroke="orange" strokeWidth="2.5" strokeDasharray="6 4" />
              )}

              {/* X-Axis labels */}
              {periods.map((p, i) => {
                const showYear = new Set(periods.map(item => item.year)).size > 1;
                const label = showYear ? `${p.month} ${p.year}` : p.month;
                const labelY = HEIGHT - PADDING + 40;
                const lx = PADDING + (i + 0.5) * groupSpacing;
                return (
                  <text key={i} x={lx} y={labelY} fontSize="8" textAnchor="end" transform={`rotate(-45 ${lx} ${labelY})`}>
                    {label}
                  </text>
                );
              })}
            </svg>
          </div>

          <div className="mt-3">
            <div className="d-flex justify-content-between align-items-center">
              <div>
                <strong>X-Axis:</strong> Months &nbsp;&nbsp;
                <strong>Y-Axis:</strong> Sales Amount
              </div>
              <div>
                <span style={{ color: "blue", fontSize: "14px" }}>●</span> <strong>Actual</strong>
                &nbsp;&nbsp;
                <span style={{ color: "green", fontSize: "14px" }}>●</span> <strong>Predicted</strong>
                &nbsp;&nbsp;
                <span style={{ color: "red", fontSize: "14px" }}>—</span> <strong>Actual Trend</strong>
                &nbsp;&nbsp;
                <span style={{ color: "orange", fontSize: "14px" }}>—</span> <strong>Predicted Trend</strong>
              </div>
            </div>
          </div>
        </>
      )}

      {/* PIE CHART */}
      {chartType === "pie" && (
        <div className="d-flex">
          <svg width="200" height="200" viewBox="0 0 200 200">
            {pieSlices.map((s, i) => (
              <path key={i} d={s.path} fill={s.color} />
            ))}
          </svg>

          <div className="ms-3">
            <strong>Legend</strong>
            {pieSlices.map((s, i) => (
              <div key={i} className="d-flex align-items-center mb-1">
                <div
                  style={{
                    width: 15,
                    height: 15,
                    backgroundColor: s.color,
                    marginRight: 8
                  }}
                />
                <small>{s.label}: {s.percentage}%</small>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* LINE GRAPH */}
      {chartType === "line" && (
        <>
          <div style={{ overflowX: "auto", overflowY: "visible", width: "100%", maxWidth: "100%", paddingBottom: "20px" }}>
            <svg width={WIDTH} height={HEIGHT + 80} style={{ display: "block", minWidth: "100%" }}>
            {/* Axes */}
            <line x1={PADDING} y1={HEIGHT - PADDING} x2={WIDTH - PADDING} y2={HEIGHT - PADDING} stroke="black" />
            <line x1={PADDING} y1={PADDING} x2={PADDING} y2={HEIGHT - PADDING} stroke="black" />

              {/* Y-Axis ticks (more readings - every 10%) */}
              {[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((ratio, i) => {
              const value = Math.round(maxValue * ratio);
              const y = HEIGHT - PADDING - value * scaleY;
              return (
                <g key={i}>
                    <line x1={PADDING - 5} y1={y} x2={PADDING} y2={y} stroke="black" strokeWidth={ratio === 0 || ratio === 0.5 || ratio === 1.0 ? "1.5" : "0.5"} />
                    <text x={PADDING - 15} y={y + 4} fontSize="9" textAnchor="end">{value}</text>
                </g>
              );
            })}

              {/* X-Axis labels - show every month */}
              {sortedData.map((d, i) => {
                const showYear = new Set(sortedData.map(item => item.year)).size > 1;
                const label = showYear ? `${d.month} ${d.year}` : d.month;
                const labelY = HEIGHT - PADDING + 40;
                return (
              <text
                key={i}
                x={PADDING + i * scaleX}
                    y={labelY}
                    fontSize="8"
                    textAnchor="end"
                    transform={`rotate(-45 ${PADDING + i * scaleX} ${labelY})`}
              >
                    {label}
              </text>
                );
              })}

              {/* Plot lines based on available data */}
              {current.length > 0 && drawLine(current, "blue")}
              {predicted.length > 0 && drawLine(predicted, "green")}
          </svg>
          </div>

          <div className="mt-3">
            <div className="d-flex justify-content-between align-items-center">
              <div>
            <strong>X-Axis:</strong> Months &nbsp;&nbsp;
            <strong>Y-Axis:</strong> Sales Amount
              </div>
              <div>
                {current.length > 0 && (
                  <>
                    <span style={{ color: "blue", fontSize: "14px" }}>●</span> <strong>Actual Sales</strong>
                    {predicted.length > 0 && <>&nbsp;&nbsp;</>}
                  </>
                )}
                {predicted.length > 0 && (
                  <>
                    <span style={{ color: "green", fontSize: "14px" }}>●</span> <strong>Predicted Sales</strong>
                  </>
                )}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default SalesChart;
