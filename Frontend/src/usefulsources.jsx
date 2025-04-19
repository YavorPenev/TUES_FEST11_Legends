import React from "react";
import { Link } from "react-router";

function UsefulSources() {
    return (
        <div className="w-screen justify-center items-center flex text-blue-50 mt-5">
            <div className="p-6 bg-none max-w-max flex flex-col gap-2 justify-center items-center bg-blue-900 mx-7 mb-10 rounded-2xl">
                <h1 className="font-bold text-4xl ">Useful Sources</h1>
                <div className="flex flex-row gap-6 justify-center items-center">
                    <img src="/usefulsources.png" className="h-28 w-28" />
                    <h3 className="font-semibold text-2xl">Here are some useful sources to brighten your business skills and inform yourself.</h3>
                </div>
                <Link to="/usefulsources" className="p-2 font-bold bg-blue-100 text-blue-950 rounded-xl text-2xl transition-transform hover:scale-110 active:duration-50 active:scale-90 hover:font-extrabold">Useful Sources</Link>


            </div>
        </div>
    );
}

export default UsefulSources;