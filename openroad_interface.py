"""
OpenROAD Interface for PPAForge AI
Python wrapper for OpenROAD EDA tool integration.

This module provides a clean interface to OpenROAD for:
- Running placement algorithms
- Extracting design metrics
- Applying placement updates
- Running timing/power analysis
"""

import os
import subprocess
import re
import tempfile
import json
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OpenROADInterface:
    """
    Interface to OpenROAD EDA tool.
    
    Provides methods to:
    1. Run default OpenROAD placement flow
    2. Apply custom placements from RL agent
    3. Extract PPA metrics
    4. Run incremental optimization
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary with OpenROAD settings
        """
        self.config = config
        self.openroad_config = config.get('openroad', {})
        self.design_config = config.get('design', {})
        self.pdk_config = config.get('pdk', {})
        
        # OpenROAD binary and paths
        self.openroad_bin = self.openroad_config.get('binary_path', 'openroad')
        self.flow_scripts_path = self.openroad_config.get('flow_scripts_path', '/tools/OpenROAD-flow-scripts')
        self.platform = self.openroad_config.get('platform', 'sky130hd')
        
        # Verify OpenROAD installation
        self._verify_installation()
        
        # Working directory for OpenROAD runs
        self.work_dir = tempfile.mkdtemp(prefix='ppaforge_openroad_')
        
        logger.info(f"OpenROAD Interface initialized with platform: {self.platform}")
        logger.info(f"Work directory: {self.work_dir}")
    
    def _verify_installation(self):
        """Verify OpenROAD is installed and accessible."""
        try:
            result = subprocess.run(
                [self.openroad_bin, '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"OpenROAD found: {result.stdout.strip()}")
            else:
                logger.warning("OpenROAD binary found but version check failed")
        except FileNotFoundError:
            logger.error(
                f"OpenROAD not found at {self.openroad_bin}. "
                "Please install OpenROAD or update the path in config."
            )
        except Exception as e:
            logger.error(f"Error verifying OpenROAD: {e}")
    
    def run_default_placement(
        self,
        design_path: str,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run default OpenROAD placement flow.
        
        This serves as the baseline for comparison.
        
        Args:
            design_path: Path to design files
            output_dir: Output directory for results
        
        Returns:
            Dictionary with placement results and metrics
        """
        if output_dir is None:
            output_dir = os.path.join(self.work_dir, 'baseline')
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Running default OpenROAD placement for {design_path}")
        
        # Create OpenROAD script
        script_path = self._create_placement_script(design_path, output_dir)
        
        # Run OpenROAD
        try:
            result = subprocess.run(
                [self.openroad_bin, script_path],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                cwd=output_dir
            )
            
            if result.returncode != 0:
                logger.error(f"OpenROAD failed: {result.stderr}")
                return {'success': False, 'error': result.stderr}
            
            logger.info("OpenROAD placement completed successfully")
            
            # Parse results
            placement = self._parse_placement_result(output_dir)
            metrics = self._extract_metrics(output_dir)
            
            return {
                'success': True,
                'placement': placement,
                'metrics': metrics,
                'output_dir': output_dir
            }
            
        except subprocess.TimeoutExpired:
            logger.error("OpenROAD timed out")
            return {'success': False, 'error': 'Timeout'}
        except Exception as e:
            logger.error(f"Error running OpenROAD: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_placement_script(
        self,
        design_path: str,
        output_dir: str
    ) -> str:
        """
        Create Tcl script for OpenROAD placement.
        
        Args:
            design_path: Path to design files
            output_dir: Output directory
        
        Returns:
            Path to generated script
        """
        script_path = os.path.join(output_dir, 'place.tcl')
        
        # Find design files
        verilog_file = self._find_file(design_path, ['.v', '.sv'])
        sdc_file = self._find_file(design_path, ['.sdc'])
        
        # Get design parameters
        clock_period = self.design_config.get('clock_period', 10.0)
        target_density = self.design_config.get('target_density', 0.65)
        
        # Generate Tcl script
        tcl_script = f"""
# OpenROAD Placement Script
# Generated by PPAForge AI

# Set platform
set platform "{self.platform}"

# Load libraries
set liberty_file "{self.pdk_config.get('lib_path', '')}/sky130_fd_sc_hd__tt_025C_1v80.lib"
set tech_lef "{self.pdk_config.get('tech_lef', '')}"
set std_cell_lef "{self.pdk_config.get('std_cell_lef', '')}"

# Read LEF files
read_lef $tech_lef
read_lef $std_cell_lef

# Read liberty
read_liberty $liberty_file

# Read verilog
read_verilog "{verilog_file}"
link_design [lindex [yosys get_attr top] 0]

# Read SDC
if {{[file exists "{sdc_file}"]}} {{
    read_sdc "{sdc_file}"
}}

# Initialize floorplan
initialize_floorplan \\
    -die_area "0 0 1000 1000" \\
    -core_area "10 10 990 990" \\
    -site unithd

# Place IO pins
place_pins -hor_layers metal3 -ver_layers metal2

# Global placement
global_placement \\
    -density {target_density} \\
    -timing_driven

# Detailed placement
detailed_placement

# Optimize placement
optimize_placement

# Write results
write_def "{output_dir}/placement.def"

# Report metrics
report_checks -path_delay max -format summary > "{output_dir}/timing.rpt"
report_wns > "{output_dir}/wns.rpt"
report_tns > "{output_dir}/tns.rpt"

# Report power (if available)
if {{[catch {{report_power}} error]}} {{
    puts "Power analysis not available: $error"
}} else {{
    report_power > "{output_dir}/power.rpt"
}}

# Exit
exit
"""
        
        with open(script_path, 'w') as f:
            f.write(tcl_script)
        
        logger.info(f"Created OpenROAD script: {script_path}")
        
        return script_path
    
    def _find_file(self, directory: str, extensions: List[str]) -> str:
        """Find file with given extensions in directory."""
        for ext in extensions:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith(ext):
                        return os.path.join(root, file)
        return ""
    
    def _parse_placement_result(self, output_dir: str) -> Dict:
        """
        Parse placement DEF file to extract cell positions.
        
        Args:
            output_dir: Directory containing placement.def
        
        Returns:
            Dictionary mapping cell names to positions
        """
        def_file = os.path.join(output_dir, 'placement.def')
        
        if not os.path.exists(def_file):
            logger.warning(f"Placement DEF not found: {def_file}")
            return {}
        
        placement = {}
        
        with open(def_file, 'r') as f:
            content = f.read()
        
        # Parse components
        component_pattern = r'-\s+(\S+)\s+\S+\s+.*?PLACED\s*\(\s*(\d+)\s+(\d+)\s*\)\s*(\S+)'
        
        for match in re.finditer(component_pattern, content):
            cell_name = match.group(1)
            x = int(match.group(2))
            y = int(match.group(3))
            orientation = match.group(4)
            
            placement[cell_name] = {
                'x': x,
                'y': y,
                'orientation': orientation,
                'rotation': self._orientation_to_rotation(orientation)
            }
        
        logger.info(f"Parsed {len(placement)} cell placements")
        
        return placement
    
    def _orientation_to_rotation(self, orientation: str) -> int:
        """Convert DEF orientation to rotation index."""
        orientation_map = {
            'N': 0,
            'E': 1,
            'S': 2,
            'W': 3,
            'FN': 0,
            'FE': 1,
            'FS': 2,
            'FW': 3
        }
        return orientation_map.get(orientation, 0)
    
    def _extract_metrics(self, output_dir: str) -> Dict:
        """
        Extract PPA metrics from OpenROAD reports.
        
        Args:
            output_dir: Directory containing report files
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'power': 0.0,
            'dynamic_power': 0.0,
            'leakage_power': 0.0,
            'worst_slack': 0.0,
            'total_negative_slack': 0.0,
            'wirelength': 0.0,
            'area': 0.0,
            'runtime': 0.0
        }
        
        # Parse timing report
        wns_file = os.path.join(output_dir, 'wns.rpt')
        if os.path.exists(wns_file):
            with open(wns_file, 'r') as f:
                content = f.read()
                match = re.search(r'wns\s+([-\d.]+)', content)
                if match:
                    metrics['worst_slack'] = float(match.group(1))
        
        tns_file = os.path.join(output_dir, 'tns.rpt')
        if os.path.exists(tns_file):
            with open(tns_file, 'r') as f:
                content = f.read()
                match = re.search(r'tns\s+([-\d.]+)', content)
                if match:
                    metrics['total_negative_slack'] = float(match.group(1))
        
        # Parse power report
        power_file = os.path.join(output_dir, 'power.rpt')
        if os.path.exists(power_file):
            with open(power_file, 'r') as f:
                content = f.read()
                
                # Total power
                match = re.search(r'Total\s+Power:\s+([\d.]+)\s*([a-zA-Z]+)', content)
                if match:
                    power = float(match.group(1))
                    unit = match.group(2)
                    # Convert to watts
                    if unit == 'mW':
                        power *= 1e-3
                    elif unit == 'uW':
                        power *= 1e-6
                    metrics['power'] = power
                
                # Dynamic power
                match = re.search(r'Dynamic\s+Power:\s+([\d.]+)\s*([a-zA-Z]+)', content)
                if match:
                    power = float(match.group(1))
                    unit = match.group(2)
                    if unit == 'mW':
                        power *= 1e-3
                    elif unit == 'uW':
                        power *= 1e-6
                    metrics['dynamic_power'] = power
                
                # Leakage power
                match = re.search(r'Leakage\s+Power:\s+([\d.]+)\s*([a-zA-Z]+)', content)
                if match:
                    power = float(match.group(1))
                    unit = match.group(2)
                    if unit == 'mW':
                        power *= 1e-3
                    elif unit == 'uW':
                        power *= 1e-6
                    metrics['leakage_power'] = power
        
        logger.info(f"Extracted metrics: {metrics}")
        
        return metrics
    
    def get_default_placement(self) -> Dict:
        """
        Get default placement from OpenROAD.
        
        Returns:
            Dictionary of cell placements
        """
        # This would run a quick placement and return results
        # For now, return empty dict (would be populated in real use)
        logger.info("Getting default placement from OpenROAD")
        return {}
    
    def apply_custom_placement(
        self,
        placement: Dict,
        design_path: str,
        output_dir: str
    ) -> Dict:
        """
        Apply custom placement from RL agent and evaluate.
        
        Args:
            placement: Cell placements from agent
            design_path: Path to design files
            output_dir: Output directory
        
        Returns:
            Results and metrics
        """
        logger.info(f"Applying custom placement with {len(placement)} cells")
        
        # Create DEF file with custom placement
        def_path = os.path.join(output_dir, 'custom_placement.def')
        self._write_placement_def(placement, def_path)
        
        # Run detailed placement and optimization on custom placement
        script_path = self._create_refinement_script(
            design_path,
            def_path,
            output_dir
        )
        
        # Execute OpenROAD
        try:
            result = subprocess.run(
                [self.openroad_bin, script_path],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=output_dir
            )
            
            if result.returncode != 0:
                logger.error(f"OpenROAD refinement failed: {result.stderr}")
                return {'success': False}
            
            # Extract metrics
            metrics = self._extract_metrics(output_dir)
            
            return {
                'success': True,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error applying custom placement: {e}")
            return {'success': False, 'error': str(e)}
    
    def _write_placement_def(self, placement: Dict, output_path: str):
        """Write placement to DEF file."""
        # This would generate a proper DEF file
        # Simplified implementation
        logger.info(f"Writing placement DEF to {output_path}")
        
        # In real implementation, this would create a valid DEF file
        # with COMPONENTS section containing placement coordinates
    
    def _create_refinement_script(
        self,
        design_path: str,
        def_path: str,
        output_dir: str
    ) -> str:
        """Create script for refining custom placement."""
        script_path = os.path.join(output_dir, 'refine.tcl')
        
        tcl_script = f"""
# Refinement script for custom placement

# Read DEF with custom placement
read_def "{def_path}"

# Run detailed placement
detailed_placement

# Optimize
optimize_placement

# Write results
write_def "{output_dir}/refined_placement.def"

# Report metrics
report_checks -path_delay max -format summary > "{output_dir}/timing.rpt"
report_wns > "{output_dir}/wns.rpt"
report_power > "{output_dir}/power.rpt"

exit
"""
        
        with open(script_path, 'w') as f:
            f.write(tcl_script)
        
        return script_path
    
    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
            logger.info(f"Cleaned up work directory: {self.work_dir}")


def test_openroad_interface():
    """Test OpenROAD interface."""
    import yaml
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    interface = OpenROADInterface(config)
    
    print(f"OpenROAD binary: {interface.openroad_bin}")
    print(f"Platform: {interface.platform}")
    print(f"Work directory: {interface.work_dir}")
    
    # Cleanup
    interface.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_openroad_interface()
