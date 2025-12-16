import React from "react";

const Navbar = ({ onSelect }) => {
  return (
    <nav className="navbar navbar-dark bg-dark px-4">
      <span className="navbar-brand">Sales Prediction System</span>

      <select
        className="form-select w-auto"
        onChange={(e) => onSelect(e.target.value)}
      >
        <option value="actual">Actual Sales</option>
        <option value="analysis">This Year Analysis</option>
        <option value="1year">Next 1 Year Prediction</option>
        <option value="2year">Next 2 Years Prediction</option>
      </select>
    </nav>
  );
};

export default Navbar;
