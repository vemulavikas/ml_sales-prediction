import React from "react";

const SalesTable = ({ data }) => {
  if (!data || data.length === 0) return null;

  return (
    <table className="table table-bordered table-striped">
      <thead>
        <tr>
          <th>Month</th>
          <th>Year</th>
          <th>Amount</th>
          <th>Type</th>
        </tr>
      </thead>
      <tbody>
        {data.map((row, index) => (
          <tr key={index}>
            <td>{row.month}</td>
            <td>{row.year}</td>
            <td>{row.amount}</td>
            <td>{row.type}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default SalesTable;
