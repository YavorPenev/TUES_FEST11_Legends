import React from 'react';
import Header from './assets/header';
import Footer from './assets/footer';

function Law() {
  return (
    <>
      <Header />
      <div className="p-5 pt-24 bg-gradient-to-br from-blue-600 to-blue-200">
        <h1 className=" text-4xl font-bold mb-10 text-center mt-10 text-blue-200">
          -- Laws of Finances in the Republic of Bulgaria --
        </h1>

        <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg w-3/4 mx-auto">
          <h2 className="text-2xl font-bold text-blue-500 mb-3 text-center">
            Ministry of Finance of the Republic of Bulgaria
          </h2>
          <p className="text-gray-700 mb-3 text-left">
            The Ministry of Finance of the Republic of Bulgaria is responsible for developing and implementing the country's financial policy. Its main role includes managing public finances, preparing the state budget, regulating the tax system, and overseeing the enforcement of financial laws. The Ministry also ensures compliance with European Union financial regulations and works to maintain economic stability in the country.
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

        <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg w-3/4 mx-auto">
          <h2 className="text-2xl font-bold text-blue-500 mb-3 text-center">
            Banking Act
          </h2>
          <p className="text-gray-700 mb-3 text-left">
            The Banking Act in Bulgaria regulates the activities of credit institutions, ensuring financial stability, transparency, and consumer protection. It sets guidelines for licensing, supervision, and risk management in the banking sector. The law also establishes rules for capital adequacy, liquidity, and the prevention of systemic risks in the banking system. It aims to protect depositors and ensure the smooth functioning of the financial system.
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

        <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg w-3/4 mx-auto">
          <h2 className="text-2xl font-bold text-blue-500 mb-3 text-center">
            Consumer Protection in Financial Services Act
          </h2>
          <p className="text-gray-700 mb-3 text-left">
            This law protects consumers engaging in financial services such as banking, insurance, loans, and investments. It ensures fair treatment, clear and accurate information, and safeguards consumers' financial rights. The law also requires financial institutions to disclose all terms and conditions transparently, preventing misleading practices. It provides mechanisms for resolving disputes between consumers and financial service providers.
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

        <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg w-3/4 mx-auto">
          <h2 className="text-2xl font-bold text-blue-500 mb-3 text-center">
            Anti-Money Laundering Act
          </h2>
          <p className="text-gray-700 mb-3 text-left">
            The Anti-Money Laundering Act (AML) is designed to prevent money laundering and the financing of terrorism. It requires financial institutions to implement compliance measures such as customer due diligence, transaction monitoring, and reporting suspicious activities. The law also establishes penalties for non-compliance and provides guidelines for international cooperation in combating financial crimes.
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

        <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg w-3/4 mx-auto">
          <h2 className="text-2xl font-bold text-blue-500 mb-3 text-center">
            Investment Protection Act
          </h2>
          <p className="text-gray-700 mb-3 text-left">
            The Investment Protection Act ensures fair and transparent conditions for investors in Bulgaria. It provides legal protection against unfair practices and establishes dispute resolution mechanisms for investment-related conflicts. The law also promotes foreign investments by offering guarantees and incentives to international investors, fostering economic growth and development.
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

        <section className="mb-10 bg-blue-200 p-6 rounded-lg shadow-lg w-3/4 mx-auto">
          <h2 className="text-2xl font-bold text-blue-500 mb-3 text-center">
            Financial Markets and Securities Law
          </h2>
          <p className="text-gray-700 mb-3 text-left">
            This law regulates Bulgaria’s financial markets, ensuring the fair trading of securities, protecting investors, and preventing financial fraud. It outlines rules for stock exchanges, brokers, and financial instruments. The law also establishes transparency requirements for publicly traded companies and provides mechanisms for monitoring and enforcing compliance with market regulations.
          </p>
          <a
            href="https://www.fsc.bg/en/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline "
          >
            Visit the Financial Supervision Commission
          </a>
        </section>
      </div>
      <Footer />
    </>
  );
}

export default Law;