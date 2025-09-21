# import google.generativeai as genai
# import os
# import logging
# from dotenv import load_dotenv

# # Load variables from the .env file into the environment
# load_dotenv()

# # --- Configuration ---
# # Get the key from the environment
# gemini_api_key = os.environ.get('GEMINI_API_KEY')

# # Check if the key was found
# if not gemini_api_key:
#     # If not, log an error and STOP the script
#     logging.error("FATAL: GEMINI_API_KEY not found in environment variables.")
#     exit() 

# # >>> THIS IS THE MISSING STEP <<<
# # Give the key to the genai library
# genai.configure(api_key=gemini_api_key)


# # --- Model Initialization ---
# # Now this will work because the library has been configured
# model = genai.GenerativeModel('gemini-1.0-pro')

# def search_news(topic: str):
#     """
#     Uses the Gemini API to search for news on a given topic.
#     """
#     print(f"\n🔍 Searching for news about: '{topic}'...")

#     prompt = (
#         f"Act as a news search engine. Your task is to find and summarize the 3 most recent and relevant "
#         f"news articles about the following topic: '{topic}'. "
#         f"For each article, provide:\n"
#         f"1. A clear headline.\n"
#         f"2. A concise summary (2-3 sentences).\n"
#         f"3. The primary source or publication if available.\n"
#         f"Please format the output clearly."
#     )

#     try:
#         response = model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"An error occurred: {e}"

# # --- Main Execution ---
# if __name__ == "__main__":
#     user_topic = input("Enter the news topic you want to search for: ")
    
#     if user_topic:
#         news_summary = search_news(user_topic)
#         print("\n--- Latest News Summary ---")
#         print(news_summary)
#         print("---------------------------\n")
#     else:
#         print("No topic entered. Please run the script again.")



from ddgs import DDGS

results = DDGS().text("python programming", max_results=5)
print(results)