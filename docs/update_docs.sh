# Automatic Doxygen documentation update script
# Author: Auto-generated
# Description: Updates Doxygen documentation for TieBreaker AI project

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOXYFILE_PATH="$PROJECT_DIR/docs/Doxyfile"
OUTPUT_DIR="$PROJECT_DIR/docs/api"
BACKUP_DIR="$PROJECT_DIR/docs/api_backup_$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}=== Doxygen Documentation Update ===${NC}"
echo -e "${BLUE}Project directory: $PROJECT_DIR${NC}"

if ! command -v doxygen &> /dev/null; then
    echo -e "${RED}Error: Doxygen is not installed or not found in PATH${NC}"
    echo -e "${YELLOW}To install on Ubuntu/Debian: sudo apt-get install doxygen${NC}"
    echo -e "${YELLOW}To install on Fedora/CentOS: sudo yum install doxygen${NC}"
    echo -e "${YELLOW}To install on Arch: sudo pacman -S doxygen${NC}"
    exit 1
fi

if [[ ! -f "$DOXYFILE_PATH" ]]; then
    echo -e "${RED}Error: Doxyfile not found at $DOXYFILE_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Doxygen found: $(doxygen --version)${NC}"
echo -e "${GREEN}✓ Doxyfile found: $DOXYFILE_PATH${NC}"

if [[ -d "$OUTPUT_DIR" ]]; then
    echo -e "${YELLOW}Backing up old documentation...${NC}"
    cp -r "$OUTPUT_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}✓ Backup created: $BACKUP_DIR${NC}"
fi

if [[ -d "$OUTPUT_DIR" ]]; then
    echo -e "${YELLOW}Cleaning old documentation...${NC}"
    rm -rf "$OUTPUT_DIR"/*
fi

echo -e "${YELLOW}Generating Doxygen documentation...${NC}"
cd "$PROJECT_DIR"

if doxygen "$DOXYFILE_PATH"; then
    echo -e "${GREEN}✓ Documentation generated successfully!${NC}"
    
    if [[ -d "$OUTPUT_DIR/html" ]]; then
        HTML_FILES=$(find "$OUTPUT_DIR/html" -name "*.html" | wc -l)
        echo -e "${GREEN}✓ $HTML_FILES HTML files generated${NC}"
        echo -e "${GREEN}✓ Documentation available at: $OUTPUT_DIR/html/index.html${NC}"
        
        # Automatically open documentation in browser (optional)
        if [[ "$1" == "--open" ]] && command -v xdg-open &> /dev/null; then
            echo -e "${BLUE}Opening documentation in browser...${NC}"
            xdg-open "$OUTPUT_DIR/html/index.html" &
        fi
    fi
    
    if [[ -d "$BACKUP_DIR" ]]; then
        echo -e "${YELLOW}Removing backup (generation successful)...${NC}"
        rm -rf "$BACKUP_DIR"
    fi
    
    echo -e "${GREEN}=== Update completed successfully! ===${NC}"
    
else
    echo -e "${RED}Error during documentation generation${NC}"
    
    if [[ -d "$BACKUP_DIR" ]]; then
        echo -e "${YELLOW}Restoring backup...${NC}"
        rm -rf "$OUTPUT_DIR"
        mv "$BACKUP_DIR" "$OUTPUT_DIR"
        echo -e "${GREEN}✓ Old documentation restored${NC}"
    fi
    
    exit 1
fi