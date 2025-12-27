# `files_db` Reimplementation Plan

This document outlines the plan to reimplement the legacy `files_db` system using the modern `mongoODM` framework.

## Part 1: Enhance the `mongoODM` Framework

Before reimplementing `files_db`, we will add critical features to the core `mongoODM` framework to support the requirements discovered during the analysis phase.

### Step 1.1: Declarative Compound Indexes

*   **Goal**: Allow developers to define compound and text indexes directly in their `Document` schemas for better performance and maintainability.
*   **Implementation**:
    1.  Modify `metaclass.py`: The `DocumentMetaclass` will be updated to read an `indexes` list from a `Meta` inner class within a `Document`.
    2.  The `indexes` list will support tuples for compound indexes (e.g., `[('field1', 1), ('field2', -1)]`) and dictionaries for text or other complex indexes.
    3.  Modify `db_component.py`: The `_create_all_indexes` method in `MongoODMComponent` will be updated to parse this `indexes` list and create the corresponding indexes in MongoDB upon application startup.
*   **Example**:
    ```python
    class MyDocument(Document):
        field1 = StringField()
        field2 = StringField()

        class Meta:
            collection = "my_documents"
            indexes = [
                [('field1', 1), ('field2', -1)],
                {'fields': ['$**'], 'name': 'text_index'}
            ]
    ```

### Step 1.2: Generic Reference Field

*   **Goal**: Implement a `GenericReferenceField` that can create a reference to a document in *any* collection, mimicking the flexible linking capability of the old system.
*   **Implementation**:
    1.  Create a new `GenericReferenceField` class in `fields.py`.
    2.  This field will store a dictionary containing the `_id` and the `_cls` (collection name) of the referenced document.
    3.  A custom getter will be implemented to automatically resolve this reference, fetching the correct document from the correct collection and returning an instantiated ODM object.

### Step 1.3: Testing and Documentation

*   **Goal**: Ensure the new framework features are robust, reliable, and easy for other developers to use.
*   **Implementation**:
    1.  Update `tests/test_odm.py` with new unit tests specifically for compound indexes and the `GenericReferenceField`.
    2.  Update the `README.md` in the `mongoODM` directory to document these powerful new features with clear explanations and code examples.

## Part 2: Reimplement `files_db` Data Models

With the enhanced framework in place, we will define the new ODM schemas for the `files_db` system.

### Step 2.1: Define Core Schemas

*   **`CollectionItem`**: An abstract base document (`abstract=True`) with common fields like `rating`, `description`, `tags`, etc.
*   **`FileRecord`**: Inherits from `CollectionItem`. Will include fields for path components, metadata, and a `ListField` of `EmbeddedDocumentField(AIExpertise)`.
*   **`AIExpertise`**: An `EmbeddedDocument` to store the results from indexers like LLaVA.
*   **`TagRecord`**: A document for tags, with a `ReferenceField` to a parent tag to create the hierarchy.
*   **`RelationRecord`**: A document with two `GenericReferenceField`s (`from_doc` and `to_doc`) to link any two documents.
*   **`DetectionObjectClass`**, **`Detection`**, **`RecognizedObject`**: A set of documents for the object recognition system, using inheritance and `ReferenceField`s to link them.
*   **`AnnotationJob`** and **`AnnotationRecord`**: Documents for the annotation system.

## Part 3: Refactor Business Logic into Services

All business logic will be moved out of the data models and into dedicated service classes.

*   **`IndexingService`**: This service will manage the entire "AI Expertise" pipeline. It will take a `FileRecord`, use a `FileTypeRouter` to determine the correct indexers to run (e.g., `FaceDetector`, `DeepDanBoru`, `MetadataReader`), and orchestrate the process.
*   **`TagService`**: Will contain all logic for managing tags, including creating, renaming, and remapping them.
*   **`AnnotationService`**: Will manage the creation of annotation jobs, the process of annotating items, and the exporting of datasets.
*   **`FileService`**: Will contain logic for file-related operations, such as adding all files from a folder to the database.

## Part 4: Finalization

*   **Update `files_db_module.py`**: The main module file will be updated to use the new services and the `MongoODMComponent`.
*   **Create Real-World Examples**: A new example script will be created to demonstrate the usage of the new `files_db` system, including creating and indexing a file, tagging it, and running an annotation job.
*   **Delete Old Code**: Once the new system is verified, the old `mongoext` wrappers and other legacy components will be deleted.
