import axios from "axios";

const API_URL = "http://127.0.0.1:5000";

export const getActualSales = async () => {
  const res = await axios.get(`${API_URL}/actual`);
  return res.data;
};

export const getThisYearAnalysis = async () => {
  const res = await axios.get(`${API_URL}/analysis`);
  return res.data;
};

export const getSalesForecast = async (months) => {
  const res = await axios.get(`${API_URL}/forecast`, {
    params: { months },
  });
  return res.data;
};
