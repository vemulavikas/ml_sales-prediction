// Actual sales data
export const actualSales = [
  { month: "Jan", year: 2024, amount: 12000, type: "Actual" },
  { month: "Feb", year: 2024, amount: 15000, type: "Actual" },
  { month: "Mar", year: 2024, amount: 18000, type: "Actual" },
  { month: "Apr", year: 2024, amount: 16000, type: "Actual" },
  { month: "May", year: 2024, amount: 20000, type: "Actual" },
  { month: "Jun", year: 2024, amount: 22000, type: "Actual" },
  { month: "Jul", year: 2024, amount: 21000, type: "Actual" },
  { month: "Aug", year: 2024, amount: 23000, type: "Actual" },
  { month: "Sep", year: 2024, amount: 24000, type: "Actual" },
  { month: "Oct", year: 2024, amount: 26000, type: "Actual" },
  { month: "Nov", year: 2024, amount: 25000, type: "Actual" },
  { month: "Dec", year: 2024, amount: 27000, type: "Actual" },
];

// This year analysis
export const currentYearAnalysis = {
  year: 2024,
  total_sales: 250000,
  average_sales: 20833,
  highest_month: "December",
  lowest_month: "January",
  monthly_sales: actualSales,
};

// Next 1 year prediction
export const nextYearPrediction = [
  { month: "Jan", year: 2025, amount: 28000, type: "Predicted" },
  { month: "Feb", year: 2025, amount: 29000, type: "Predicted" },
  { month: "Mar", year: 2025, amount: 30000, type: "Predicted" },
  { month: "Apr", year: 2025, amount: 31000, type: "Predicted" },
  { month: "May", year: 2025, amount: 32000, type: "Predicted" },
  { month: "Jun", year: 2025, amount: 33000, type: "Predicted" },
  { month: "Jul", year: 2025, amount: 34000, type: "Predicted" },
  { month: "Aug", year: 2025, amount: 35000, type: "Predicted" },
  { month: "Sep", year: 2025, amount: 36000, type: "Predicted" },
  { month: "Oct", year: 2025, amount: 37000, type: "Predicted" },
  { month: "Nov", year: 2025, amount: 38000, type: "Predicted" },
  { month: "Dec", year: 2025, amount: 39000, type: "Predicted" },
];

// Next 2 year prediction
export const twoYearPrediction = [
  ...nextYearPrediction,
  { month: "Jan", year: 2026, amount: 40000, type: "Predicted" },
  { month: "Feb", year: 2026, amount: 41000, type: "Predicted" },
  { month: "Mar", year: 2026, amount: 42000, type: "Predicted" },
  { month: "Apr", year: 2026, amount: 43000, type: "Predicted" },
  { month: "May", year: 2026, amount: 44000, type: "Predicted" },
  { month: "Jun", year: 2026, amount: 45000, type: "Predicted" },
  { month: "Jul", year: 2026, amount: 46000, type: "Predicted" },
  { month: "Aug", year: 2026, amount: 47000, type: "Predicted" },
  { month: "Sep", year: 2026, amount: 48000, type: "Predicted" },
  { month: "Oct", year: 2026, amount: 49000, type: "Predicted" },
  { month: "Nov", year: 2026, amount: 50000, type: "Predicted" },
  { month: "Dec", year: 2026, amount: 51000, type: "Predicted" },
];
