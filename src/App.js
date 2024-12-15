import React, { useState } from "react";
import axios from "axios";

function App() {
  const [imageSrc, setImageSrc] = useState(null);
  const [currentPage, setCurrentPage] = useState(1); // For paginated APIs
  const [currentApi, setCurrentApi] = useState("");
  const [loading, setLoading] = useState(false); // State for loading animation

  const fetchPaginatedImage = async (url, page) => {
    try {
      setLoading(true); // Start loading animation
      setImageSrc(null);
      const response = await axios.get(`${url}?page=${page}`, { responseType: "blob" });
      const imageBlob = new Blob([response.data], { type: "image/png" });
      const imageUrl = URL.createObjectURL(imageBlob);
      setImageSrc(imageUrl);
    } catch (error) {
      console.error("Error fetching image:", error);
      alert("Error fetching image!");
    } finally {
      setLoading(false); // Stop loading animation
    }
  };

  const handleApiCall = (url) => {
    setCurrentPage(1);
    setCurrentApi(url);
    fetchPaginatedImage(url, 1);
  };

  const handleNextPage = () => {
    const nextPage = currentPage + 1;
    fetchPaginatedImage(currentApi, nextPage);
    setCurrentPage(nextPage);
  };

  const handlePreviousPage = () => {
    if (currentPage > 1) {
      const previousPage = currentPage - 1;
      fetchPaginatedImage(currentApi, previousPage);
      setCurrentPage(previousPage);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif", textAlign: "center" }}>
      <h1 style={{ color: "#34495E", marginBottom: "30px" }}>E-Commerce Data Analysis</h1>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", justifyContent: "center", marginBottom: "30px" }}>
        <button
          onClick={() => handleApiCall("http://127.0.0.1:5000/api/total-sales")}
          style={buttonStyle}
        >
          Total Sales
        </button>
        <button
          onClick={() => handleApiCall("http://127.0.0.1:5000/api/avg-order-by-value")}
          style={buttonStyle}
        >
          Average Order by Value
        </button>
        <button
          onClick={() => handleApiCall("http://127.0.0.1:5000/api/customer-segmentation")}
          style={buttonStyle}
        >
          Customer Segmentation
        </button>
        <button
          onClick={() => handleApiCall("http://127.0.0.1:5000/api/performance")}
          style={buttonStyle}
        >
          Performance
        </button>
        <button
          onClick={() => handleApiCall("http://127.0.0.1:5000/api/customer-loyalty")}
          style={buttonStyle}
        >
          Customer Loyalty
        </button>
        <button
          onClick={() => handleApiCall("http://127.0.0.1:5000/api/shipping-analysis")}
          style={buttonStyle}
        >
          Shipping Analysis
        </button>
        <button
          onClick={() => handleApiCall("http://127.0.0.1:5000/api/forecast")}
          style={buttonStyle}
        >
          Sales Forecast
        </button>
      </div>

      {loading && (
        <div style={{ margin: "30px auto" }}>
          <div
            style={{
              border: "4px solid #f3f3f3",
              borderTop: "4px solid #3498db",
              borderRadius: "50%",
              width: "40px",
              height: "40px",
              animation: "spin 1s linear infinite",
              margin: "0 auto",
            }}
          ></div>
          <style>
            {`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
            `}
          </style>
        </div>
      )}

      {imageSrc && (
        <div style={{ marginTop: "20px" }}>
          <h2 style={{ color: "#2C3E50" }}>Visualization</h2>
          <img src={imageSrc} alt="Visualization" style={{ width: "80%", maxHeight: "600px", margin: "0 auto" }} />
          {(currentApi === "http://127.0.0.1:5000/api/total-sales" ||
            currentApi === "http://127.0.0.1:5000/api/avg-order-by-value") && (
            <div style={{ marginTop: "20px", display: "flex", justifyContent: "center", gap: "10px" }}>
              <button onClick={handlePreviousPage} disabled={currentPage === 1} style={paginationButtonStyle}>
                Previous
              </button>
              <button onClick={handleNextPage} style={paginationButtonStyle}>
                Next
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const buttonStyle = {
  backgroundColor: "#3498db",
  color: "#fff",
  padding: "10px 20px",
  border: "none",
  borderRadius: "5px",
  cursor: "pointer",
  transition: "background-color 0.3s",
  fontSize: "14px",
};

const paginationButtonStyle = {
  ...buttonStyle,
  backgroundColor: "#1abc9c",
};

buttonStyle[":hover"] = {
  backgroundColor: "#2980b9",
};

paginationButtonStyle[":hover"] = {
  backgroundColor: "#16a085",
};

export default App;
