#!/bin/bash

# BrainMod Step 2 - Master Demo Runner
# This script orchestrates and runs all BrainMod Step 2 demonstration scripts

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Demo configuration
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$DEMO_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
RESULTS_DIR="$PROJECT_DIR/results"
TOTAL_START_TIME=$(date +%s)

# Create necessary directories
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_DIR/demo.log"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_DIR/demo.log"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_DIR/demo.log"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_DIR/demo.log"
}

log_section() {
    echo -e "\n${PURPLE}=======================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}=======================================${NC}" | tee -a "$LOG_DIR/demo.log"
}

# Function to display header
display_header() {
    clear
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║            BrainMod Step 2 - Master Demo Runner             ║"
    echo "║                                                              ║"
    echo "║        Comprehensive Demonstration of Multi-Expert           ║"
    echo "║              Brain Simulation with STDP Learning            ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
}

# Function to check prerequisites
check_prerequisites() {
    log_section "Checking Prerequisites"
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check if virtual environment exists
    if [ ! -d "$PROJECT_DIR/brainmod-env" ]; then
        log_warning "Virtual environment not found. Running setup..."
        cd "$PROJECT_DIR"
        bash "$DEMO_DIR/setup-dev.sh"
        cd "$DEMO_DIR"
    fi
    
    # Check if BrainMod modules are available
    if [ ! -d "$PROJECT_DIR/brainmod" ]; then
        log_error "BrainMod source code not found at $PROJECT_DIR/brainmod"
        missing_deps+=("brainmod source")
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies and try again."
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Function to run a single demo
run_demo() {
    local demo_script="$1"
    local demo_name="$2"
    local demo_start_time=$(date +%s)
    local demo_log="$LOG_DIR/${demo_script%.sh}.log"
    local demo_result="$RESULTS_DIR/${demo_script%.sh}_result.txt"
    
    log_section "Running: $demo_name"
    
    echo -e "${CYAN}Script:${NC} $demo_script"
    echo -e "${CYAN}Log:${NC} $demo_log"
    echo -e "${CYAN}Result:${NC} $demo_result"
    echo ""
    
    # Run the demo with error handling
    if cd "$PROJECT_DIR" && source "brainmod-env/bin/activate" && bash "$DEMO_DIR/$demo_script" > "$demo_log" 2>&1; then
        local demo_end_time=$(date +%s)
        local demo_duration=$((demo_end_time - demo_start_time))
        
        log_success "$demo_name completed successfully (Duration: ${demo_duration}s)"
        
        # Store result summary
        echo "Demo: $demo_name" > "$demo_result"
        echo "Status: SUCCESS" >> "$demo_result"
        echo "Duration: ${demo_duration}s" >> "$demo_result"
        echo "Log: $demo_log" >> "$demo_result"
        echo "Completed: $(date)" >> "$demo_result"
        
        return 0
    else
        local demo_end_time=$(date +%s)
        local demo_duration=$((demo_end_time - demo_start_time))
        
        log_error "$demo_name failed (Duration: ${demo_duration}s)"
        
        # Store error summary
        echo "Demo: $demo_name" > "$demo_result"
        echo "Status: FAILED" >> "$demo_result"
        echo "Duration: ${demo_duration}s" >> "$demo_result"
        echo "Log: $demo_log" >> "$demo_result"
        echo "Failed: $(date)" >> "$demo_result"
        echo "Error: Check log file for details" >> "$demo_result"
        
        return 1
    fi
}

# Function to run demo sequence
run_demo_sequence() {
    log_section "Demo Execution Sequence"
    
    local demos=(
        "step2_demo.sh:Core Step 2 Features"
        "quick_demo.sh:Quick Start Demo"
        "affect_demo.sh:Affect Detection Demo"
        "performance_demo.sh:Performance Optimization Demo"
        "advanced_demo.sh:Advanced Multi-Expert Demo"
        "auto_learning_demo.sh:STDP Learning Demo"
    )
    
    local successful_demos=()
    local failed_demos=()
    
    for demo_spec in "${demos[@]}"; do
        IFS=':' read -r demo_script demo_name <<< "$demo_spec"
        
        # Check if demo script exists
        if [ ! -f "$DEMO_DIR/$demo_script" ]; then
            log_warning "Demo script not found: $demo_script"
            failed_demos+=("$demo_name (Script not found)")
            continue
        fi
        
        # Run the demo
        if run_demo "$demo_script" "$demo_name"; then
            successful_demos+=("$demo_name")
        else
            failed_demos+=("$demo_name")
        fi
        
        echo ""
    done
    
    # Display results
    log_section "Demo Results Summary"
    
    echo -e "${GREEN}Successful Demos (${#successful_demos[@]}):${NC}"
    for demo in "${successful_demos[@]}"; do
        echo -e "  ${GREEN}✓${NC} $demo"
    done
    echo ""
    
    if [ ${#failed_demos[@]} -gt 0 ]; then
        echo -e "${RED}Failed Demos (${#failed_demos[@]}):${NC}"
        for demo in "${failed_demos[@]}"; do
            echo -e "  ${RED}✗${NC} $demo"
        done
        echo ""
    fi
    
    return $(( ${#failed_demos[@]} == 0 ? 0 : 1 ))
}

# Function to generate comprehensive report
generate_report() {
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - TOTAL_START_TIME))
    local report_file="$RESULTS_DIR/comprehensive_demo_report.txt"
    
    log_section "Generating Comprehensive Report"
    
    cat > "$report_file" << EOF
================================================================================
                    BRAINMOD STEP 2 - COMPREHENSIVE DEMO REPORT
================================================================================

Report Generated: $(date)
Total Execution Time: ${total_duration}s
Demo Directory: $DEMO_DIR
Project Directory: $PROJECT_DIR
Log Directory: $LOG_DIR
Results Directory: $RESULTS_DIR

================================================================================
                              SYSTEM INFORMATION
================================================================================

Operating System: $(uname -a)
Python Version: $(python3 --version 2>&1 || echo "Not available")
Available Memory: $(free -h | grep '^Mem:' | awk '{print $2}' || echo "Not available")
Available CPU Cores: $(nproc || echo "Not available")
Disk Usage: $(df -h . | grep '^/dev/' | awk '{print $4}' | head -1 || echo "Not available")

================================================================================
                              DEMO EXECUTION LOGS
================================================================================

EOF
    
    # Add individual demo logs
    for log_file in "$LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            echo "--- $(basename "$log_file") ---" >> "$report_file"
            echo "" >> "$report_file"
            cat "$log_file" >> "$report_file"
            echo "" >> "$report_file"
            echo "================================================================================
" >> "$report_file"
        fi
    done
    
    # Add demo results
    echo "

================================================================================
                             INDIVIDUAL DEMO RESULTS
================================================================================

" >> "$report_file"
    
    for result_file in "$RESULTS_DIR"/*_result.txt; do
        if [ -f "$result_file" ]; then
            echo "--- $(basename "$result_file" _result.txt) ---" >> "$report_file"
            echo "" >> "$report_file"
            cat "$result_file" >> "$report_file"
            echo "" >> "$report_file"
            echo "================================================================================
" >> "$report_file"
        fi
    done
    
    log_success "Comprehensive report generated: $report_file"
    
    # Also create a summary JSON file for programmatic access
    local json_report="$RESULTS_DIR/demo_summary.json"
    cat > "$json_report" << EOF
{
    "report_generated": "$(date -Iseconds)",
    "total_duration": $total_duration,
    "demo_directory": "$DEMO_DIR",
    "project_directory": "$PROJECT_DIR",
    "log_directory": "$LOG_DIR",
    "results_directory": "$RESULTS_DIR",
    "system_info": {
        "os": "$(uname -a)",
        "python_version": "$(python3 --version 2>&1 || echo "Not available")",
        "memory": "$(free -h | grep '^Mem:' | awk '{print $2}' || echo "Not available")",
        "cpu_cores": $(nproc || echo "null"),
        "disk_available": "$(df -h . | grep '^/dev/' | awk '{print $4}' | head -1 || echo "Not available")"
    },
    "demo_results": {
EOF
    
    # Add individual demo results to JSON
    local first=true
    for result_file in "$RESULTS_DIR"/*_result.txt; do
        if [ -f "$result_file" ]; then
            local demo_name=$(basename "$result_file" _result.txt)
            local status=$(grep "Status:" "$result_file" | cut -d: -f2 | tr -d ' ')
            local duration=$(grep "Duration:" "$result_file" | cut -d: -f2 | tr -d 's ')
            
            if [ "$first" = false ]; then
                echo "," >> "$json_report"
            fi
            first=false
            
            echo "        \"$demo_name\": {" >> "$json_report"
            echo "            \"status\": \"$status\"," >> "$json_report"
            echo "            \"duration\": $duration" >> "$json_report"
            echo -n "        }" >> "$json_report"
        fi
    done
    
    echo "" >> "$json_report"
    echo "    }" >> "$json_report"
    echo "}" >> "$json_report"
    
    log_success "JSON summary report generated: $json_report"
}

# Function to display final summary
display_final_summary() {
    local total_end_time=$(date +%s)
    local total_duration=$((total_end_time - TOTAL_START_TIME))
    
    log_section "Demo Run Complete!"
    
    echo -e "${CYAN}Total Execution Time:${NC} ${total_duration}s"
    echo -e "${CYAN}Logs Directory:${NC} $LOG_DIR"
    echo -e "${CYAN}Results Directory:${NC} $RESULTS_DIR"
    echo ""
    
    echo -e "${GREEN}Key Outputs:${NC}"
    echo -e "  ${GREEN}•${NC} Detailed logs in: $LOG_DIR/"
    echo -e "  ${GREEN}•${NC} Demo results in: $RESULTS_DIR/"
    echo -e "  ${GREEN}•${NC} Comprehensive report: $RESULTS_DIR/comprehensive_demo_report.txt"
    echo -e "  ${GREEN}•${NC} JSON summary: $RESULTS_DIR/demo_summary.json"
    echo ""
    
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Review the comprehensive report for detailed analysis"
    echo "2. Check individual demo logs for specific insights"
    echo "3. Examine demo results for performance metrics"
    echo "4. Use JSON summary for programmatic analysis"
    echo ""
    
    echo -e "${CYAN}Individual Demo Access:${NC}"
    echo "Run any demo individually:"
    echo "  ./step2_demo.sh           # Core Step 2 features"
    echo "  ./quick_demo.sh           # Quick start demo"
    echo "  ./affect_demo.sh          # Affect detection"
    echo "  ./performance_demo.sh     # Performance optimization"
    echo "  ./advanced_demo.sh        # Advanced multi-expert"
    echo "  ./auto_learning_demo.sh   # STDP learning"
    echo ""
    
    log_success "All demos completed! Check the results directory for detailed output."
}

# Function to handle cleanup on exit
cleanup() {
    local exit_code=$?
    log_info "Cleaning up..."
    
    # Deactivate virtual environment if active
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null || true
    fi
    
    exit $exit_code
}

# Set up trap for cleanup
trap cleanup EXIT

# Main execution function
main() {
    # Display header
    display_header
    
    # Initialize log file
    echo "BrainMod Step 2 - Master Demo Run" > "$LOG_DIR/demo.log"
    echo "Started: $(date)" >> "$LOG_DIR/demo.log"
    echo "================================" >> "$LOG_DIR/demo.log"
    echo "" >> "$LOG_DIR/demo.log"
    
    # Check prerequisites
    check_prerequisites
    
    # Ask user for confirmation
    echo ""
    echo -e "${YELLOW}This will run all BrainMod Step 2 demonstration scripts.${NC}"
    echo -e "${YELLOW}The process may take several minutes and will generate detailed logs.${NC}"
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Demo run cancelled by user"
        exit 0
    fi
    
    # Run all demos
    run_demo_sequence
    local demo_status=$?
    
    # Generate comprehensive report
generate_report
    
    # Display final summary
    display_final_summary
    
    if [ $demo_status -eq 0 ]; then
        log_success "All demos completed successfully!"
    else
        log_warning "Some demos failed. Check logs for details."
    fi
    
    return $demo_status
}

# Run main function
main "$@"