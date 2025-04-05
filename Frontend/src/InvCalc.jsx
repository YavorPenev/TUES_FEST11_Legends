import React from 'react';
import Header from './header';
import Footer from './footer';

function InvestCalc() {
  return (
    <>
      <Header />
      <div className="p-5 pt-24 bg-gray-100">
        <h1 className="text-blue-600 text-4xl font-bold mb-10 text-center">
          -- InvestCalc --
        </h1>

      </div>
      <Footer />
    </>
  );
}

export default InvestCalc;