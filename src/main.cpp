#include "Config.hpp"
#include "Logger.hpp"

#include <CLI/CLI.hpp>

#include <stdexcept>
#include <string>


int main(int argc, char** argv) {
    try {
        CLI::App app{"RAG"};


        std::string build_path;
        std::string chat_path;


        app.add_option("--build", build_path, "Build vector database");

        app.add_option("--chat", chat_path, "Chat with RAG system");


        CLI11_PARSE(app, argc, argv);


        if(!build_path.empty() && !chat_path.empty())
            throw std::runtime_error("Cannot use --build and --chat together");


        if(build_path.empty() && chat_path.empty())
            throw std::runtime_error("Please specify --build <path> or --chat <path>");


        Config config("config.json");


        if(!build_path.empty()) {
            Logger::info("Running build mode");
            Logger::info("Input path: " + build_path);


            // TODO:
            // Load documents
            // Split documents into chunks
            // Generate embeddings
            // Build vector database
        }
        else {
            Logger::info("Running chat mode");
            Logger::info("Input path: " + chat_path);


            // TODO:
            // Load vector database
            // Read questions
            // Retrieve relevant documents
            // Generate answers
        }
    }
    catch(const std::exception& e) {
        Logger::error(e.what());
        return -1;
    }


    return 0;
}