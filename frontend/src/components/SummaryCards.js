import React from "react";

const SummaryCards = ({ data }) => {
  if (!data) return null;

  return (
    <div className="row my-3">
      <div className="col-md-3">
        <div className="card text-center bg-light">
          <h6>Total Sales</h6>
          <h4>{data.total_sales}</h4>
        </div>
      </div>
      <div className="col-md-3">
        <div className="card text-center bg-light">
          <h6>Average Sales</h6>
          <h4>{data.average_sales}</h4>
        </div>
      </div>
      <div className="col-md-3">
        <div className="card text-center bg-light">
          <h6>Highest Month</h6>
          <h4>{data.highest_month}</h4>
        </div>
      </div>
      <div className="col-md-3">
        <div className="card text-center bg-light">
          <h6>Lowest Month</h6>
          <h4>{data.lowest_month}</h4>
        </div>
      </div>
    </div>
  );
};

export default SummaryCards;
