mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
\n\
[theme]\n\
primaryColor = '#1E88E5'\n\
backgroundColor = '#FFFFFF'\n\
secondaryBackgroundColor = '#F0F2F6'\n\
textColor = '#000000'\n\
" > ~/.streamlit/config.toml
