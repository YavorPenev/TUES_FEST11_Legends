import React from 'react';
import Header from './header';
import Footer from './footer';

function Law() {
  return (
    <>
    <Header/>
    <div className="p-5 pt-24 bg-gray-100">
      <h1 className="text-blue-600 text-4xl font-bold mb-10 text-center">
        -- Laws of Finances in the Republic of Bulgaria --
      </h1>


      <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold text-blue-500 mb-3">
          Ministry of Finance of the Republic of Bulgaria
        </h2>
        <p className="text-gray-700 mb-3">
          The Ministry of Finance of the Republic of Bulgaria is responsible for developing and implementing the country's financial policy. Its main role includes managing public finances, preparing the state budget, regulating the tax system, and overseeing the enforcement of financial laws.
        </p>
        <a
          href="https://www.minfin.bg/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
          Visit the Ministry of Finance Website
        </a>
      </section>


      <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold text-blue-500 mb-3">Banking Act</h2>
        <p className="text-gray-700 mb-3">
          The Banking Act in Bulgaria regulates the activities of credit institutions, ensuring financial stability, transparency, and consumer protection. It sets guidelines for licensing, supervision, and risk management in the banking sector.
        </p>
        <a
          href="https://bnb.bg/bnbweb/groups/public/documents/bnb_law/laws_bnb_new_en.pdf"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
          Read the Banking Act (PDF)
        </a>
      </section>


      <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold text-blue-500 mb-3">
          Consumer Protection in Financial Services Act
        </h2>
        <p className="text-gray-700 mb-3">
          This law protects consumers engaging in financial services such as banking, insurance, loans, and investments. It ensures fair treatment, clear and accurate information, and safeguards consumers' financial rights.
        </p>
        <a
          href="https://commission.europa.eu/document/download/c999dfc1-d17e-4542-bd7e-b0006805e9ec_en?filename=national-consumer-organisations_bg_listing.pdf"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
          Learn More About Consumer Protection
        </a>
      </section>


      <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold text-blue-500 mb-3">Anti-Money Laundering Act</h2>
        <p className="text-gray-700 mb-3">
          The Anti-Money Laundering Act (AML) is designed to prevent money laundering and the financing of terrorism. It requires financial institutions to implement compliance measures such as customer due diligence and transaction monitoring.
        </p>
        <a
          href="https://www.dato.bg/docs/Measures_Against_Money_Laundering_Act-%20%D0%97%D0%B0%D0%BA%D0%BE%D0%BD%20%D0%B7%D0%B0%20%D0%BC%D0%B5%D1%80%D0%BA%D0%B8%D1%82%D0%B5%20%D1%81%D1%80%D0%B5%D1%89%D1%83%20%D0%B8%D0%B7%D0%BF%D0%B8%D1%80%D0%B0%D0%BD%D0%B5%D1%82%D0%BE%20%D0%BD%D0%B0%20%D0%BF%D0%B0%D1%80%D0%B8.pdf"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
          Read the Anti-Money Laundering Act (PDF)
        </a>
      </section>


      <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold text-blue-500 mb-3">Investment Protection Act</h2>
        <p className="text-gray-700 mb-3">
          The Investment Protection Act ensures fair and transparent conditions for investors in Bulgaria. It provides legal protection against unfair practices and establishes dispute resolution mechanisms for investment-related conflicts.
        </p>
        <a
          href="https://www.csi.ca/en/learning/courses/bfia"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
          Learn More About Investment Protection
        </a>
      </section>


      <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg">
        <h2 className="text-2xl font-bold text-blue-500 mb-3">Financial Markets and Securities Law</h2>
        <p className="text-gray-700 mb-3">
          This law regulates Bulgaria’s financial markets, ensuring the fair trading of securities, protecting investors, and preventing financial fraud. It outlines rules for stock exchanges, brokers, and financial instruments.
        </p>
        <a
          href="https://www.fsc.bg/en/"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 hover:underline"
        >
          Visit the Financial Supervision Commission
        </a>
      </section>
    </div>
    <Footer/>
    </>
  );
}

export default Law;