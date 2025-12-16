import React, { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import SalesChart from "../components/SalesChart";
import SalesTable from "../components/SalesTable";
import SummaryCards from "../components/SummaryCards";

import {
  getActualSales,
  getThisYearAnalysis,
  getSalesForecast,
} from "../api/api";

const Dashboard = () => {
  const [data, setData] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [title, setTitle] = useState("");
  const [mode, setMode] = useState("actual");

  useEffect(() => {
    const loadData = async () => {
      if (mode === "actual") {
        const res = await getActualSales();
        setData(res.data);
        setAnalysis(null);
        setTitle("Actual Sales");
      }

      if (mode === "analysis") {
        const res = await getThisYearAnalysis();
        setAnalysis(res.analysis);
        setData([]);
        setTitle("This Year Analysis");
      }

      if (mode === "1year") {
        const res = await getSalesForecast(12);
        setData(
          res.predictions.map((p) => {
            const d = new Date(p.date);
            return {
              month: d.toLocaleString("default", { month: "short" }),
              year: d.getFullYear(),
              amount: Math.round(p.forecast),
              type: "Predicted",
            };
          })
        );
        setAnalysis(null);
        setTitle("Next 1 Year Prediction");
      }

      if (mode === "2year") {
        const res = await getSalesForecast(24);
        setData(
          res.predictions.map((p) => {
            const d = new Date(p.date);
            return {
              month: d.toLocaleString("default", { month: "short" }),
              year: d.getFullYear(),
              amount: Math.round(p.forecast),
              type: "Predicted",
            };
          })
        );
        setAnalysis(null);
        setTitle("Next 2 Years Prediction");
      }
    };

    loadData();
  }, [mode]);

  return (
    <>
      <Navbar onSelect={setMode} />

      <div className="container mt-4">
        {analysis && <SummaryCards data={analysis} />}

        {data.length > 0 && (
          <div className="row">
            <div className="col-md-7 mb-4">
              <SalesChart data={data} title={title} />
            </div>
            <div className="col-md-5 mb-4">
              <SalesTable data={data} />
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default Dashboard;
