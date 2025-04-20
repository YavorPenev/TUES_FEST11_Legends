import React from "react";
import { Link } from "react-router";


function LawLink() {
    return (
        <div className="mt-3 w-full max-w-screen-lg mx-auto px-4 mb-6">
            <Link
                to="/law"
                className="cursor-pointer rounded-2xl overflow-hidden shadow-md hover:shadow-xl transition transform hover:scale-105 bg-blue-900 text-white flex flex-col md:flex-row items-center p-6"
            >
                <img
                    src="./lawlink.png"
                    alt="Investments and Finance"
                    className="w-full md:w-1/2 h-48 md:h-64 object-cover rounded-3xl mb-4 md:mb-0 md:mr-6"
                />

                <div className="flex-1">
                    <h2 className="text-xl md:text-2xl font-bold mb-2">
                        Explore Financial & Investment Regulations
                    </h2>
                    <p className="text-sm md:text-base text-left indent-4">
                        In this section, you’ll find important information about financial regulations, investment laws,
                        and market guidelines that influence both personal and corporate finance. Learn about the rules
                        that protect investors, regulate financial institutions, and shape investment opportunities.
                        Whether you’re a beginner or experienced investor, these resources will help you navigate
                        the legal side of finance with confidence.
                        <b className="text-1xl"> Click here</b> to explore the full list of sources and legal insights.
                    </p>
                </div>
            </Link>
        </div>
    );
};

export default LawLink;