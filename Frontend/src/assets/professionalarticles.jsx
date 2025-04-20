import React from "react";
import { Link } from "react-router";


function ProfArticles() {
  return (
    <div className="mt-5 w-full max-w-screen-lg mx-auto px-4">
      <Link
        to="/PerArticles"
        className="cursor-pointer rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition transform hover:scale-105 bg-blue-900 text-white flex flex-col md:flex-row items-center p-6"
      >
    
        <img
          src="./learn.png" 
          alt="Investments and Finance"
          className="w-full md:w-1/2 h-48 md:h-64 object-cover rounded-3xl mb-4 md:mb-0 md:mr-6"
        />

        <div className="flex-1">
          <h2 className="text-xl md:text-2xl font-bold mb-2">
            Discover the World of Investments and Financial Strategies
          </h2>
          <p className="text-sm md:text-base text-left indent-4">
            Dive into a comprehensive collection of articles written by
            experienced professionals in the fields of investments, financial
            planning, and wealth management. These articles provide strategies
            to achieve financial independence and grow your wealth. Whether
            you're a beginner or an experienced investor, you'll find valuable
            insights and tips to help you make informed decisions.{" "}
            <b className="text-base">Click here</b> to explore and start your journey toward financial
            success!
          </p>
        </div>
      </Link>
    </div>
  );
}

export default ProfArticles;