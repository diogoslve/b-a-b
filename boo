ServiceNow Authorization Analysis, Capability Mapping, Simulation, and ATF Builder Platform

Objective

Design and implement a web-based platform that analyzes, explains, simulates, and validates user permissions and business capabilities in a ServiceNow instance.

The purpose of the platform is to support a large-scale role and authorization redesign effort in a heavily customized ServiceNow environment where the current security model is difficult to understand.

The platform should help answer:

* What can a user actually do?
* Why can they do it?
* Which groups, roles, ACLs, scripts, and configurations contribute to that capability?
* What would happen if a role, group assignment, ACL, or configuration were changed?
* Which permissions are actually exercised in the platform?
* How can we automatically validate assumptions through ATF?

The platform should prioritize explainability, traceability, impact analysis, and test-driven validation.

⸻

Technical Constraints

The solution must use only:

* JavaScript
* HTML
* CSS
* PowerShell
* SQLite
* ServiceNow REST APIs

Assume:

* No server-side frameworks
* No package installation requirements
* No external databases
* No Node.js dependencies unless they can run without installation
* SQLite is the only persistence layer

⸻

Architecture

ServiceNow Instance
↓
REST API Extraction
↓
PowerShell Data Collection
↓
SQLite Database
↓
HTML / CSS / JavaScript Application

The application should function as a self-contained analysis environment.

⸻

Data Extraction Requirements

Extract and maintain a local model of the following ServiceNow data.

Identity and Access

* sys_user
* sys_user_role
* sys_user_has_role
* sys_user_group
* sys_user_grmember
* sys_group_has_role

ACLs

* sys_security_acl
* sys_security_acl_role

Capture:

* operation
* type
* field
* condition
* script
* active status

⸻

Table Hierarchy

* sys_db_object

Capture inheritance relationships.

⸻

Security Logic

Identify and inventory:

* Scripted ACLs
* ACL conditions
* Script Includes referenced by ACLs
* Dynamic role checks

⸻

Business Capability Logic

Inventory and analyze:

Business Rules

Capture:

* table
* condition
* script
* active status

Determine whether they:

* abort actions
* modify records
* enforce process restrictions

⸻

UI Actions

Capture:

* condition
* script
* role requirements

Determine which actions users can actually execute.

Examples:

* Resolve Incident
* Close Incident
* Approve Change
* Cancel Change

⸻

Flow Designer Flows

Capture:

* trigger conditions
* approval logic
* user restrictions
* decision points

⸻

Data Policies

Capture:

* mandatory fields
* read-only enforcement
* conditions

⸻

UI Policies

Capture:

* field visibility
* read-only states
* mandatory states

Document but assign lower security importance.

⸻

Script Includes

Inventory all Script Includes referenced by:

* ACLs
* Business Rules
* UI Actions
* Flows

Build dependency mapping.

⸻

Domain Separation

If enabled:

Capture domain relationships and visibility restrictions.

⸻

Authorization Model

Build a graph model.

User
→ Group
→ Role
→ ACL
→ Table
→ Permission

Support traceability.

Example output:

John Smith
→ Service Desk Group
→ itil
→ incident.write
→ Incident
→ Update

⸻

Effective Permission Engine

Calculate:

* Direct roles
* Group-derived roles
* Inherited roles

Generate:

Effective User Permissions

Store provenance for every permission.

Example:

Permission:
incident.write

Source:
Service Desk Group
→ itil
→ ACL

⸻

Capability-Based Analysis

Do not focus only on roles.

Model business capabilities.

Examples:

Incident Management

* Read Incident
* Create Incident
* Update Incident
* Assign Incident
* Resolve Incident
* Close Incident
* Reopen Incident
* Delete Incident

Change Management

* Read Change
* Create Change
* Update Change
* Submit Change
* Approve Change
* Reject Change
* Implement Change
* Close Change

CMDB

* Read CI
* Create CI
* Modify CI
* Delete CI

Knowledge

* Read Article
* Create Article
* Publish Article
* Retire Article

Service Catalog

* Submit Request
* Approve Request
* Fulfill Request
* Cancel Request

Capabilities should be mapped to the underlying technical implementation.

⸻

Persona Modeling

Support persona definitions.

Examples:

* Requester
* Service Desk L1
* Service Desk L2
* Incident Manager
* Change Coordinator
* CAB Approver
* Service Owner
* Knowledge Manager
* CMDB Administrator

Map:

Persona
→ Groups
→ Roles
→ Permissions
→ Capabilities

⸻

State-Based Capability Analysis

Permissions must be evaluated across states.

Example: Incident

Action	New	In Progress	Resolved	Closed
Read	?	?	?	?
Edit	?	?	?	?
Resolve	?	?	?	?
Reopen	?	?	?	?

Example: Change

Action	New	Assess	Authorize	Scheduled	Implement	Review	Closed
Read	?	?	?	?	?	?	?
Edit	?	?	?	?	?	?	?
Approve	?	?	?	?	?	?	?

⸻

Impact Simulation Engine

Support simulations.

Examples:

Remove Role

Remove Group Assignment

Remove ACL

Modify ACL

Modify Group Membership

Outputs:

* affected users
* affected groups
* affected capabilities
* permissions lost
* permissions gained

⸻

Security Cleanup Analysis

Automatically identify:

Orphaned Roles

Roles with:

* no users
* no groups

⸻

Unused Roles

Roles never contributing to effective permissions.

⸻

Duplicate Roles

Roles granting identical permission sets.

⸻

Excessive Privilege

Users receiving permissions from multiple redundant sources.

⸻

Toxic Combinations

Examples:

Requester + Approver

Requester + Fulfillment

Identify segregation-of-duties concerns.

⸻

Capability Dependency Mapping

Build dependency trees.

Example:

Resolve Incident

Depends On:

ACL
incident.write

UI Action
Resolve

Business Rule
Validate Resolution

Script Include
CanResolve()

Flow
Post Resolution Notifications

Show every dependency involved in a capability.

⸻

ATF Builder Module

The platform must include a visual ATF Builder.

The purpose is to generate ATF tests from capabilities and personas.

⸻

Block-Based Builder

Implement a drag-and-drop or block-based builder.

Example:

START
↓
Impersonate User
↓
Open Record
↓
Set Field Value
↓
Click UI Action
↓
Validate Success
↓
END

Blocks should be reusable.

⸻

Supported Block Types

User Context

* Impersonate User
* End Impersonation
* Set Persona

Navigation

* Open Table
* Open Record
* Open Form
* Open List

Data

* Create Record
* Update Record
* Delete Record
* Set Field
* Validate Field

Actions

* Click UI Action
* Submit Form
* Approve Record
* Reject Record
* Assign Record

Validation

* Expect Success
* Expect Failure
* Expect Error
* Validate State
* Validate ACL Restriction
* Validate Visibility

Logic

* If
* Else
* Loop
* Variable
* Parameter

⸻

Capability-to-ATF Generation

The system should generate tests automatically.

Example:

Capability

Resolve Incident

Generates

Impersonate L2 User
→ Open Incident
→ Change State to Resolved
→ Save
→ Validate Success

Negative Test

Impersonate L1 User
→ Open Incident
→ Change State to Resolved
→ Save
→ Validate Failure

⸻

ATF Templates

Support reusable templates.

Examples:

Incident Permission Test

Change Approval Test

Catalog Request Test

Knowledge Publishing Test

CMDB Modification Test

⸻

ATF Export

Support generation of:

* ATF record definitions
* Table API payloads
* JSON exports
* Import scripts

Allow creation of ATF records through ServiceNow APIs where supported.

⸻

Test Execution Tracking

Store:

* test runs
* execution history
* pass/fail status
* execution timestamps

Track results by:

* persona
* capability
* release version

⸻

Dashboard Requirements

Provide dashboards for:

Access Explorer

Why does this user have this permission?

⸻

Capability Explorer

What can this persona do?

⸻

Impact Simulator

What changes if something is removed?

⸻

Security Risk Dashboard

Show:

* orphaned roles
* duplicate roles
* toxic combinations
* high-risk ACLs
* scripted ACLs

⸻

ATF Dashboard

Show:

* generated tests
* execution results
* failures
* coverage percentage

⸻

Deliverables

Produce:

1. SQLite schema
2. PowerShell extraction framework
3. ServiceNow API integration layer
4. Effective permission engine
5. Capability engine
6. Dependency mapping engine
7. Role impact simulator
8. Persona manager
9. Block-based ATF builder
10. ATF export framework
11. HTML/CSS/JavaScript UI
12. Security cleanup analytics
13. Dashboard and reporting system

The solution should be designed as a ServiceNow authorization digital twin that explains why permissions exist, predicts the impact of changes, maps technical permissions to business capabilities, and validates those capabilities through automated ATF generation and execution.
